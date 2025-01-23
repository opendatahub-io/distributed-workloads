def train_func():
    import os
    import torch
    import torch.distributed as dist
    import torch.nn as nn
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, DistributedSampler

    # Initialize distributed process group
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    # Configuration
    batch_size = 64
    epochs = 5
    learning_rate = 0.01

    # Dataset and DataLoader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = datasets.MNIST(root="/tmp/datasets/mnist", train=True, download=True, transform=transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, sampler=train_sampler)

    # Model, Loss, and Optimizer
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ).cuda(local_rank)

    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    criterion = nn.CrossEntropyLoss().cuda(local_rank)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(local_rank, non_blocking=True), target.cuda(local_rank, non_blocking=True)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Log epoch stats
        print(f"Rank {rank} | Epoch {epoch + 1}/{epochs} | Loss: {epoch_loss / len(train_loader)}")

    # Cleanup
    dist.destroy_process_group()

def train_func_2():
    import os
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DistributedSampler
    from torchvision import datasets, transforms
    import torch.distributed as dist

    # [1] Setup PyTorch DDP. Distributed environment will be set automatically by Training Operator.
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    Distributor = torch.nn.parallel.DistributedDataParallel
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    print(
        "Distributed Training for WORLD_SIZE: {}, RANK: {}, LOCAL_RANK: {}".format(
            dist.get_world_size(),
            dist.get_rank(),
            local_rank,
        )
    )

    # [2] Create PyTorch CNN Model.
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = torch.nn.Conv2d(1, 20, 5, 1)
            self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
            self.fc1 = torch.nn.Linear(4 * 4 * 50, 500)
            self.fc2 = torch.nn.Linear(500, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    # [3] Attach model to the correct GPU device and distributor.
    device = torch.device(f"cuda:{local_rank}")
    model = Net().to(device)
    model = Distributor(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # [4] Setup FashionMNIST dataloader and distribute data across PyTorchJob workers.
    dataset = datasets.FashionMNIST(
        "./data",
        download=True,
        train=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=128,
        sampler=DistributedSampler(dataset),
    )

    # [5] Start model Training.
    for epoch in range(3):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Attach Tensors to the device.
            data = data.to(device)
            target = target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0 and dist.get_rank() == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tloss={:.4f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )

def train_func_3():
    import os

    import torch
    import requests
    from pytorch_lightning import LightningModule, Trainer
    from pytorch_lightning.callbacks.progress import TQDMProgressBar
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, random_split, RandomSampler
    from torchmetrics import Accuracy
    from torchvision import transforms
    from torchvision.datasets import MNIST
    import gzip
    import shutil
    from minio import Minio


    PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")
    BATCH_SIZE = 256 if torch.cuda.is_available() else 64

    local_mnist_path = os.path.dirname(os.path.abspath(__file__))

    print("prior to running the trainer")
    print("MASTER_ADDR: is ", os.getenv("MASTER_ADDR"))
    print("MASTER_PORT: is ", os.getenv("MASTER_PORT"))


    STORAGE_BUCKET_EXISTS = "{{.StorageBucketDefaultEndpointExists}}"
    print("STORAGE_BUCKET_EXISTS: ",STORAGE_BUCKET_EXISTS)
    print(f"{'Storage_Bucket_Default_Endpoint : is {{.StorageBucketDefaultEndpoint}}' if '{{.StorageBucketDefaultEndpointExists}}' == 'true' else ''}")
    print(f"{'Storage_Bucket_Name : is {{.StorageBucketName}}' if '{{.StorageBucketNameExists}}' == 'true' else ''}")
    print(f"{'Storage_Bucket_Mnist_Directory : is {{.StorageBucketMnistDir}}' if '{{.StorageBucketMnistDirExists}}' == 'true' else ''}")

    class LitMNIST(LightningModule):
        def __init__(self, data_dir=PATH_DATASETS, hidden_size=64, learning_rate=2e-4):
            super().__init__()

            # Set our init args as class attributes
            self.data_dir = data_dir
            self.hidden_size = hidden_size
            self.learning_rate = learning_rate

            # Hardcode some dataset specific attributes
            self.num_classes = 10
            self.dims = (1, 28, 28)
            channels, width, height = self.dims
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,)),
                ]
            )

            # Define PyTorch model
            self.model = nn.Sequential(
                nn.Flatten(),
                nn.Linear(channels * width * height, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size, self.num_classes),
            )

            self.val_accuracy = Accuracy()
            self.test_accuracy = Accuracy()

        def forward(self, x):
            x = self.model(x)
            return F.log_softmax(x, dim=1)

        def training_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
            preds = torch.argmax(logits, dim=1)
            self.val_accuracy.update(preds, y)

            # Calling self.log will surface up scalars for you in TensorBoard
            self.log("val_loss", loss, prog_bar=True)
            self.log("val_acc", self.val_accuracy, prog_bar=True)

        def test_step(self, batch, batch_idx):
            x, y = batch
            logits = self(x)
            loss = F.nll_loss(logits, y)
            preds = torch.argmax(logits, dim=1)
            self.test_accuracy.update(preds, y)

            # Calling self.log will surface up scalars for you in TensorBoard
            self.log("test_loss", loss, prog_bar=True)
            self.log("test_acc", self.test_accuracy, prog_bar=True)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
            return optimizer

        ####################
        # DATA RELATED HOOKS
        ####################

        def prepare_data(self):
            # download
            print("Downloading MNIST dataset...")

            if "{{.StorageBucketDefaultEndpointExists}}" == "true" and "{{.StorageBucketDefaultEndpoint}}" != "":
                print("Using storage bucket to download datasets...")
                dataset_dir = os.path.join(self.data_dir, "MNIST/raw")
                endpoint = "{{.StorageBucketDefaultEndpoint}}"
                access_key = "{{.StorageBucketAccessKeyId}}"
                secret_key = "{{.StorageBucketSecretKey}}"
                bucket_name = "{{.StorageBucketName}}"

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

            MNIST(self.data_dir, train=True, download=download_datasets)
            MNIST(self.data_dir, train=False, download=download_datasets)

        def setup(self, stage=None):

            # Assign train/val datasets for use in dataloaders
            if stage == "fit" or stage is None:
                mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
                self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage is None:
                self.mnist_test = MNIST(
                    self.data_dir, train=False, transform=self.transform
                )

        def train_dataloader(self):
            return DataLoader(self.mnist_train, batch_size=BATCH_SIZE, sampler=RandomSampler(self.mnist_train, num_samples=1000))

        def val_dataloader(self):
            return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)

        def test_dataloader(self):
            return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)


    # Init DataLoader from MNIST Dataset

    model = LitMNIST(data_dir=local_mnist_path)

    print("GROUP: ", int(os.environ.get("GROUP_WORLD_SIZE", 1)))
    print("LOCAL: ", int(os.environ.get("LOCAL_WORLD_SIZE", 1)))

    # Initialize a trainer
    trainer = Trainer(
        accelerator="has to be specified",
        # devices=1 if torch.cuda.is_available() else None,  # limiting got iPython runs
        max_epochs=3,
        callbacks=[TQDMProgressBar(refresh_rate=20)],
        num_nodes=int(os.environ.get("GROUP_WORLD_SIZE", 1)),
        devices=int(os.environ.get("LOCAL_WORLD_SIZE", 1)),
        replace_sampler_ddp=False,
        strategy="ddp",
    )

    # Train the model ⚡
    trainer.fit(model)