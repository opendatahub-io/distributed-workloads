def train_func():
    import os
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DistributedSampler
    from torchvision import datasets, transforms
    import torch.distributed as dist
    from pathlib import Path
    from minio import Minio
    import shutil
    import gzip
    from urllib.parse import urlparse

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
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model = Distributor(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    # [4] Setup FashionMNIST dataloader and distribute data across PyTorchJob workers.
    dataset_path = "./data"
    dataset_dir = os.path.join(dataset_path, "MNIST/raw")
    with_aws = "{{.StorageBucketNameExists}}"
    endpoint = "{{.StorageBucketDefaultEndpoint}}"
    access_key = "{{.StorageBucketAccessKeyId}}"
    secret_key = "{{.StorageBucketSecretKey}}"
    bucket_name = "{{.StorageBucketName}}"
    prefix = "{{.StorageBucketMnistDir}}"

    # Sanitize endpoint to remove any scheme or path.
    parsed = urlparse(endpoint)
    # If the endpoint URL contains a scheme, netloc contains the host and optional port.
    endpoint = parsed.netloc if parsed.netloc else parsed.path
    secure = parsed.scheme == "https"

    if with_aws == "true":
        client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            cert_check=False,
            secure=secure,
        )

        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)

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

        dataset = datasets.MNIST(
            dataset_path,
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        )
    else:
        dataset = datasets.MNIST(
            dataset_path,
            train=True,
            download=True,
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

    dist.barrier()
