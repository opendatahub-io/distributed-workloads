# Copyright 2023.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import gzip
import shutil
from minio import Minio

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, epoch, writer):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for batch_idx, (data, target) in enumerate(train_loader):
        # Attach tensors to the device.
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )
            niter = epoch * len(train_loader) + batch_idx
            writer.add_scalar("loss", loss.item(), niter)


def test(model, device, test_loader, writer, epoch):
    model.eval()

    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            # Attach tensors to the device.
            data, target = data.to(device), target.to(device)

            output = model(data)
            # Get the index of the max log-probability.
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    print("\naccuracy={:.4f}\n".format(float(correct) / len(test_loader.dataset)))
    writer.add_scalar("accuracy", float(correct) / len(test_loader.dataset), epoch)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch FashionMNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        metavar="M",
        help="SGD momentum (default: 0.5)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="disables CUDA training",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        metavar="S",
        help="random seed (default: 1)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    parser.add_argument(
        "--dir",
        default="logs",
        metavar="L",
        help="directory where summary logs are stored",
    )

    parser.add_argument(
        "--backend",
        type=str,
        help="Distributed backend",
        choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
        default=dist.Backend.GLOO,
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="./",
        help="Path to save the trained model",
    )

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")
        if args.backend != dist.Backend.NCCL:
            print(
                "Warning. Please use `nccl` distributed backend for the best performance using GPUs"
            )

    writer = SummaryWriter(args.dir)

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Attach model to the device.
    model = Net().to(device)

    print("Using distributed PyTorch with {} backend".format(args.backend))
    # Set distributed training environment variables to run this training script locally.
    if "WORLD_SIZE" not in os.environ:
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "1234"

    print(f"World Size: {os.environ['WORLD_SIZE']}. Rank: {os.environ['RANK']}")

    dist.init_process_group(backend=args.backend)
    model = nn.parallel.DistributedDataParallel(model)

    if all(var in os.environ for var in ["AWS_DEFAULT_ENDPOINT","AWS_ACCESS_KEY_ID","AWS_SECRET_ACCESS_KEY","AWS_STORAGE_BUCKET","AWS_STORAGE_BUCKET_MNIST_DIR"]):
        print("Using provided storage bucket to download datasets...")
        dataset_dir = os.path.join("../data/", "MNIST/raw")
        endpoint = os.environ.get("AWS_DEFAULT_ENDPOINT")
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        bucket_name = os.environ.get("AWS_STORAGE_BUCKET")
        print(f"Storage bucket endpoint: {endpoint}")
        print(f"Storage bucket name: {bucket_name}\n")

        # remove prefix if specified in storage bucket endpoint url
        secure = True
        if endpoint.startswith("https://"):
            endpoint = endpoint[len("https://") :]
        elif endpoint.startswith("http://"):
            endpoint = endpoint[len("http://") :]
            secure = False

        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            cert_check=False,
            secure=secure
        )
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        else:
            print(f"Directory '{dataset_dir}' already exists")

        # To download datasets from storage bucket's specific directory, use prefix to provide directory name
        prefix=os.environ.get("AWS_STORAGE_BUCKET_MNIST_DIR")
        print(f"Storage bucket MNIST directory prefix: {prefix}\n")

        # download all files from prefix folder of storage bucket recursively
        for item in client.list_objects(
            bucket_name, prefix=prefix, recursive=True
        ):  
            file_name=item.object_name[len(prefix)+1:]
            dataset_file_path = os.path.join(dataset_dir, file_name)
            print(f"Downloading dataset file {file_name} to {dataset_file_path}..")
            if not os.path.exists(dataset_file_path):
                client.fget_object(
                    bucket_name, item.object_name, dataset_file_path
                )
                # Unzip files -- 
                ## Sample zipfilepath : ../data/MNIST/raw/t10k-images-idx3-ubyte.gz
                with gzip.open(dataset_file_path, "rb") as f_in:
                    filename=file_name.split(".")[0]    #-> t10k-images-idx3-ubyte
                    file_path=("/".join(dataset_file_path.split("/")[:-1]))     #->../data/MNIST/raw
                    full_file_path=os.path.join(file_path,filename)     #->../data/MNIST/raw/t10k-images-idx3-ubyte
                    print(f"Extracting {dataset_file_path} to {file_path}..")

                    with open(full_file_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    print(f"Dataset file downloaded : {full_file_path}\n")
                # delete zip file
                os.remove(dataset_file_path)
            else:
                print(f"File-path '{dataset_file_path}' already exists")
        download_datasets = False
    else:
        print("Using default MNIST mirror references to download datasets ...")
        download_datasets = True

    # Get FashionMNIST train and test dataset.
    train_ds = datasets.MNIST(
        "../data",
        train=True,
        download=download_datasets,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_ds = datasets.MNIST(
        "../data",
        train=False,
        download=download_datasets,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    # Add train and test loaders.
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=DistributedSampler(train_ds),
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=args.test_batch_size,
        sampler=DistributedSampler(test_ds),
    )

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, epoch, writer)
        test(model, device, test_loader, writer, epoch)

    if args.save_model and dist.get_rank() == 0:
        model_path = os.path.join(args.output_path, "mnist_cnn.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()
