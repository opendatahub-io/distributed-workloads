def train_pytorch():
    import os
    import logging

    import torch
    from torch import nn
    import torch.nn.functional as F

    from torchvision import datasets, transforms
    import torch.distributed as dist
    from torch.utils.data import DataLoader, DistributedSampler

    # Configure logger (similar to KFTO mnist.py)
    log_formatter = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%dT%H:%M:%SZ"
    )
    logger = logging.getLogger(__file__)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)

    # [1] Configure CPU/GPU device and distributed backend.
    # Kubeflow Trainer will automatically configure the distributed environment.
    device, backend = ("cuda", "nccl") if torch.cuda.is_available() else ("cpu", "gloo")
    dist.init_process_group(backend=backend)

    local_rank = int(os.getenv("LOCAL_RANK", 0))
    logger.info(
        "Distributed Training with WORLD_SIZE: {}, RANK: {}, LOCAL_RANK: {}.".format(
            dist.get_world_size(),
            dist.get_rank(),
            local_rank,
        )
    )

    # [2] Define PyTorch CNN Model to be trained.
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
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

    # [3] Attach model to the correct device.
    device = torch.device(f"{device}:{local_rank}")
    model = nn.parallel.DistributedDataParallel(Net().to(device))
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    # [4] Get the Fashion-MNIST dataset.
    # Dataset should be pre-downloaded to avoid network dependencies.
    dataset_path = os.getenv("DATASET_PATH", "./data")
    
    # Load dataset (download=False assumes dataset is already present)
    dataset = datasets.FashionMNIST(
        dataset_path,
        train=True,
        download=False,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    # Batch size configurable via env var (smaller = more iterations = longer training)
    batch_size = int(os.getenv("BATCH_SIZE", "64"))
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset),
    )

    # [5] Define the training loop.
    num_epochs = int(os.getenv("NUM_EPOCHS", "1"))
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    for epoch in range(num_epochs):
        # Log epoch start from ALL ranks
        num_batches = len(train_loader)
        device_type = "GPU" if torch.cuda.is_available() else "CPU"
        logger.info(f"[{device_type}{global_rank}] Epoch {epoch} | Batchsize: {batch_size} | Steps: {num_batches} | World Size: {world_size}")
        
        # Set epoch for DistributedSampler to ensure proper shuffling
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        
        epoch_loss = 0.0
        num_batches_processed = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # Attach tensors to the device.
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = F.nll_loss(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track loss for epoch summary
            epoch_loss += loss.item()
            num_batches_processed += 1
            
            # Log detailed training progress from rank 0 only (to avoid log spam)
            if batch_idx % 10 == 0 and global_rank == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(inputs) * world_size,  # Adjust for distributed training
                        len(train_loader.dataset),
                        100.0 * batch_idx / num_batches,
                        loss.item(),
                    )
                )
        
        # End-of-epoch summary from ALL ranks
        avg_loss = epoch_loss / num_batches_processed
        logger.info(f"[{device_type}{global_rank}] Epoch {epoch} completed | Avg Loss: {avg_loss:.6f} | Batches: {num_batches_processed}")

    # Wait for the training to complete and destroy to PyTorch distributed process group.
    dist.barrier()
    # All ranks report completion
    logger.info(f"[{device_type}{global_rank}] Training is finished")
    dist.destroy_process_group()


if __name__ == "__main__":
    train_pytorch()