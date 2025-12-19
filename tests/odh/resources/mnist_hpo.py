import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torchvision import datasets, transforms

import ray,gzip, shutil
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import AsyncHyperBandScheduler
from minio import Minio

EPOCH_SIZE = 128
TEST_SIZE = 64

local_mnist_path = os.path.dirname(os.path.abspath(__file__))
# %%

STORAGE_BUCKET_EXISTS = "{{.StorageBucketDefaultEndpointExists}}"
print("STORAGE_BUCKET_EXISTS: ",STORAGE_BUCKET_EXISTS)
print(f"{'Storage_Bucket_Default_Endpoint : is {{.StorageBucketDefaultEndpoint}}' if '{{.StorageBucketDefaultEndpointExists}}' == 'true' else ''}")
print(f"{'Storage_Bucket_Name : is {{.StorageBucketName}}' if '{{.StorageBucketNameExists}}' == 'true' else ''}")
print(f"{'Storage_Bucket_Mnist_Directory : is {{.StorageBucketMnistDir}}' if '{{.StorageBucketMnistDirExists}}' == 'true' else ''}")


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def train_func(model, optimizer, train_loader, device=None):
    device = device or torch.device("cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test_func(model, data_loader, device=None):
    device = device or torch.device("cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def get_data_loaders(batch_size=128):
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    # download
    print("Downloading MNIST dataset...")

    if "{{.StorageBucketDefaultEndpointExists}}" == "true" and "{{.StorageBucketDefaultEndpoint}}" != "":
        print("Using storage bucket to download datasets...")
        dataset_dir = os.path.join(local_mnist_path, "MNIST/raw")
        endpoint = "{{.StorageBucketDefaultEndpoint}}"
        access_key = "{{.StorageBucketAccessKeyId}}"
        secret_key = "{{.StorageBucketSecretKey}}"
        bucket_name = "{{.StorageBucketName}}"

        #remove https prefix incase provided in endpoint url
        if endpoint.startswith("https://"):
            endpoint=endpoint[len("https://"):]

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
            secure=secure,
        )

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        else:
            print(f"Directory '{dataset_dir}' already exists")

        # To download datasets from storage bucket's specific directory, use prefix to provide directory name
        prefix="{{.StorageBucketMnistDir}}"
        # download all files from prefix folder of storage bucket recursively
        for item in client.list_objects(
            bucket_name, prefix=prefix, recursive=True
        ):  
            file_name=item.object_name[len(prefix)+1:]
            dataset_file_path = os.path.join(dataset_dir, file_name)
            print(dataset_file_path)
            if not os.path.exists(dataset_file_path):
                client.fget_object(
                    bucket_name, item.object_name, dataset_file_path
                )
            else:
                print(f"File-path '{dataset_file_path}' already exists")
            # Unzip files
            with gzip.open(dataset_file_path, "rb") as f_in:
                with open(dataset_file_path.split(".")[:-1][0], "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            # delete zip file
            os.remove(dataset_file_path)
        download_datasets = False

    else:
        print("Using default MNIST mirror reference to download datasets...")
        download_datasets = True
    
    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                local_mnist_path, train=True, download=download_datasets, transform=mnist_transforms
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                local_mnist_path, train=False, download=download_datasets, transform=mnist_transforms
            ),
            batch_size=batch_size,
            shuffle=True,
        )
    return train_loader, test_loader


def train_mnist(config):
    should_checkpoint = config.get("should_checkpoint", False)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = get_data_loaders()
    model = ConvNet().to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"]
    )

    while True:
        train_func(model, optimizer, train_loader, device)
        acc = test_func(model, test_loader, device)
        metrics = {"mean_accuracy": acc}

        # Report metrics (and possibly a checkpoint)
        if should_checkpoint:
            with tempfile.TemporaryDirectory() as tempdir:
                torch.save(model.state_dict(), os.path.join(tempdir, "model.pt"))
                train.report(metrics, checkpoint=Checkpoint.from_directory(tempdir))
        else:
            train.report(metrics)


if __name__ == "__main__":
    # for early stopping
    sched = AsyncHyperBandScheduler()
    gpu_value=int("has to be specified")
    resources_per_trial = {"cpu": 1, "gpu": gpu_value}
    tuner = tune.Tuner(
        tune.with_resources(train_mnist, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="mean_accuracy",
            mode="max",
            scheduler=sched,
            num_samples=5,
        ),
        run_config=train.RunConfig(
            name="exp",
            stop={
                "mean_accuracy": 0.98,
                "training_iteration": 5,
            },
        ),
        param_space={
            "lr": tune.loguniform(1e-4, 1e-2),
            "momentum": tune.uniform(0.1, 0.9),
        },
    )
    results = tuner.fit()

    print("Best hyperparameters config is:", results.get_best_result().config)

    assert not results.errors