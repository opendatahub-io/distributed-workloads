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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms
import os

    
def ddp_setup(backend="nccl"):
    """Setup for Distributed Data Parallel with specified backend."""
    # If CUDA is not available, use CPU as the fallback
    if torch.cuda.is_available() and backend=="nccl":
        # Check GPU availability
        num_devices = torch.cuda.device_count()
        device = int(os.environ.get("LOCAL_RANK", 0))  # Default to device 0
        if device >= num_devices:
            print(f"Warning: Invalid device ordinal {device}. Defaulting to device 0.")
            device = 0
        torch.cuda.set_device(device)
    else:
        # If no GPU is available, use Gloo backend (for CPU-only environments)
        print("No GPU available, falling back to CPU.")
        backend="gloo"
    dist.init_process_group(backend=backend)

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

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        backend: str,
    ) -> None:
        self.local_rank = int(os.environ.get("LOCAL_RANK", -1))  # Ensure fallback if LOCAL_RANK isn't set
        self.global_rank = int(os.environ["RANK"])


        self.model=model
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        self.backend = backend

        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        # Move model to the appropriate device (GPU/CPU)
        if torch.cuda.is_available() and self.backend=="nccl":
            self.device = torch.device(f'cuda:{self.local_rank}')
            self.model = DDP(self.model.to(self.device), device_ids=[self.local_rank])
        else:
            self.device=torch.device('cpu')
            self.model = DDP(self.model.to(self.device))
        print(f"Using device: {self.device}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = F.cross_entropy(output, targets)
        loss.backward()
        self.optimizer.step()

    def _run_epoch(self, epoch, backend):
        b_sz = len(next(iter(self.train_data))[0])
        if torch.cuda.is_available() and backend=="nccl":
            print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        else:
            print(f"[CPU{self.global_rank}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        if isinstance(self.train_data.sampler, DistributedSampler):
            self.train_data.sampler.set_epoch(epoch)
        for source, targets in self.train_data:
            source = source.to(self.device)
            targets = targets.to(self.device)
            self._run_batch(source, targets)

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict() if torch.cuda.is_available() else self.model.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def train(self, max_epochs: int, backend: str):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch, backend)
            if self.global_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)


def load_train_objs(lr: float):
    """Load dataset, model, and optimizer."""
    train_set = torchvision.datasets.MNIST("../data",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]))
    model = Net()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    return train_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int, useGpu: bool):
    """Prepare DataLoader with DistributedSampler."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=useGpu,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )


def main(epochs: int, save_every: int, batch_size: int, lr: float, snapshot_path: str, backend: str):
    ddp_setup(backend)
    dataset, model, optimizer = load_train_objs(lr)
    train_loader = prepare_dataloader(dataset, batch_size, torch.cuda.is_available() and backend=="nccl")
    trainer = Trainer(model, train_loader, optimizer, save_every, snapshot_path, backend)
    trainer.train(epochs, backend)
    dist.destroy_process_group()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Distributed MNIST Training")
    parser.add_argument('--epochs', type=int, required=True, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int, required=True, help='How often to save a snapshot')
    parser.add_argument('--batch_size', type=int, default=64, help='Input batch size on each device (default: 64)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--snapshot_path', type=str, default="snapshot_mnist.pt", help='Path to save snapshots (default: snapshot_mnist.pt)')
    parser.add_argument('--backend', type=str, choices=['gloo', 'nccl'], default='nccl', help='Distributed backend type (default: nccl)')
    args = parser.parse_args()

    main(
        epochs=args.epochs,
        save_every=args.save_every,
        batch_size=args.batch_size,
        lr=args.lr,
        snapshot_path=args.snapshot_path,
        backend=args.backend
    )
