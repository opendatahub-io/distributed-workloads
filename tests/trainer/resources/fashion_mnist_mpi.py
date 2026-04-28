import os
import socket
import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Subset


def train_mpi_fashion_mnist():
    rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", "-1"))
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", "0"))
    expected_size = 2
    hostname = socket.gethostname()

    assert rank >= 0, "OMPI_COMM_WORLD_RANK not set"

    # Explicitly initialize PyTorch distributed with MPI backend for this E2E.
    dist.init_process_group(backend="mpi")
    rank = dist.get_rank()
    size = dist.get_world_size()
    assert size == expected_size, f"Expected {expected_size} MPI processes, got {size}"

    formatter = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    logger = logging.getLogger("fashion_mnist_mpi")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1)
            self.fc1 = nn.Linear(9216, 128)
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(-1, 9216)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    model = Net().to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    dataset_path = os.environ.get("DATASET_PATH", "/tmp/fashion-mnist")
    max_samples = int(os.environ.get("MAX_SAMPLES", "2048"))
    batch_size = int(os.environ.get("BATCH_SIZE", "64"))
    num_epochs = int(os.environ.get("NUM_EPOCHS", "1"))

    from torchvision import datasets, transforms

    try:
        dataset = datasets.FashionMNIST(
            dataset_path,
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
        logger.info("[Rank %s/%s] Using local FashionMNIST dataset at %s", rank, size, dataset_path)
    except Exception as err:
        logger.warning(
            "[Rank %s/%s] Local FashionMNIST missing (%s). Downloading dataset.",
            rank, size, err
        )
        dataset = datasets.FashionMNIST(
            dataset_path,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
        logger.info("[Rank %s/%s] Downloaded FashionMNIST dataset to %s", rank, size, dataset_path)

    total_samples = min(len(dataset), max_samples)
    indices = list(range(rank, total_samples, size))
    rank_dataset = Subset(dataset, indices)
    train_loader = DataLoader(rank_dataset, batch_size=batch_size, shuffle=True)

    logger.info("[Rank %s/%s] Running on %s", rank, size, hostname)
    logger.info(
        "[Rank %s/%s] PyTorch %s, CUDA available: %s", rank, size, torch.__version__, use_cuda
    )

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        steps = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = F.nll_loss(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps += 1

        avg_loss = epoch_loss / steps if steps > 0 else 0.0
        logger.info("[Rank %s/%s] Epoch %s avg_loss=%.6f", rank, size, epoch, avg_loss)

    logger.info("[Rank %s/%s] MPI TrainJob test PASSED", rank, size)
    dist.destroy_process_group()


if __name__ == "__main__":
    train_mpi_fashion_mnist()
