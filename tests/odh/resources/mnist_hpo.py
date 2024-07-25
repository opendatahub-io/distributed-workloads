import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from torchvision import datasets, transforms

import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import AsyncHyperBandScheduler

EPOCH_SIZE = 128
TEST_SIZE = 64


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

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/data.lock")):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data", train=True, download=True, transform=mnist_transforms
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data", train=False, download=True, transform=mnist_transforms
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
    gpu_value="has to be specified"
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
