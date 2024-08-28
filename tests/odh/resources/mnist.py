# Copyright 2022 IBM, Red Hat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
# %%

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