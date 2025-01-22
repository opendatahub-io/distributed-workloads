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
