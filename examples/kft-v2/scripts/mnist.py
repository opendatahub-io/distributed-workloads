#!/usr/bin/env python3
def train_fashion_mnist():    
    """
    Fashion-MNIST PyTorch training script with progression tracking and checkpointing.
    """
    
    import json
    import os
    import time
    from typing import Optional
    from pathlib import Path
    import glob

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    from torch import nn
    from torch.utils.data import DataLoader, DistributedSampler
    from torchvision import datasets, transforms

    class ProgressionTracker:
        """Helper class to track and write training progression."""

        def __init__(
            self,
            total_epochs: int,
            steps_per_epoch: int,
            status_file_path: Optional[str] = None,
            update_interval: int = 30,
        ):
            """
            Initialize progression tracker.

            Args:
                total_epochs: Total number of training epochs
                steps_per_epoch: Number of steps per epoch
                status_file_path: Path where progression status will be written.
                                If None, uses TRAINJOB_PROGRESSION_FILE_PATH env var or default.
                update_interval: Minimum seconds between status updates
            """
            self.total_epochs = total_epochs
            self.steps_per_epoch = steps_per_epoch
            self.total_steps = total_epochs * steps_per_epoch
            self.status_file_path = status_file_path or os.getenv(
                "TRAINJOB_PROGRESSION_FILE_PATH", "/tmp/training_progression.json"
            )
            self.update_interval = update_interval
            self.start_time = time.time()
            self.last_update_time = 0
            self.current_epoch = 0
            self.current_step = 0
            self.metrics = {}

        def update_step(
            self,
            epoch: int,
            step: int,
            loss: float = None,
            learning_rate: float = None,
            checkpoint_dir: str = None,
            **kwargs,
        ):
            """Update current step and optionally write status."""
            # Track cumulative progress across entire training lifecycle
            # epoch is the absolute epoch number (1-based)
            # step is the current batch within the epoch (0-based)
            self.current_epoch = epoch
            self.current_step = (epoch - 1) * self.steps_per_epoch + step + 1

            # Separate optional structured training metrics and custom generic metrics
            training_metrics = {}
            generic_metrics = {}

            # Core training metrics
            if loss is not None:
                training_metrics["loss"] = str(loss)
            if learning_rate is not None:
                training_metrics["learning_rate"] = str(learning_rate)

            # Add checkpoint information if available
            if checkpoint_dir and os.path.exists(checkpoint_dir):
                try:
                    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint-') or f.startswith('epoch-')]
                    if checkpoints:
                        training_metrics["checkpoints_stored"] = len(checkpoints)
                        # Find latest checkpoint by highest number
                        def get_checkpoint_number(checkpoint_name):
                            try:
                                # Handle both checkpoint-N and epoch-N formats
                                if 'checkpoint-' in checkpoint_name:
                                    return int(checkpoint_name.split('-')[1].split('.')[0])
                                elif 'epoch-' in checkpoint_name:
                                    return int(checkpoint_name.split('-')[1].split('.')[0])
                                else:
                                    return -1
                            except (IndexError, ValueError):
                                return -1
                        
                        latest_checkpoint_name = max(checkpoints, key=get_checkpoint_number)
                        latest_checkpoint = os.path.join(checkpoint_dir, latest_checkpoint_name)
                        training_metrics["latest_checkpoint_path"] = latest_checkpoint
                except (OSError, ValueError):
                    pass

            # Process additional metrics
            for key, value in kwargs.items():
                str_value = str(value)
                
                # Map to structured TrainingMetrics fields
                if key in ['accuracy', 'train_accuracy']:
                    training_metrics["accuracy"] = str_value
                else:
                    # Everything else goes to generic metrics
                    generic_metrics[key] = str_value

            # Store metrics for status writing
            self.training_metrics = training_metrics
            self.generic_metrics = generic_metrics

            # Write status
            current_time = time.time()
            if current_time - self.last_update_time >= self.update_interval:
                message = f"Training step {self.current_step}/{self.total_steps}"
                self.write_status(message)
                self.last_update_time = current_time

        def update_epoch(self, epoch: int, checkpoint_dir: str = None, **metrics):
            """Update current epoch and write status."""
            self.current_epoch = epoch

            # Separate structured training metrics and generic metrics
            training_metrics = {}
            generic_metrics = {}

            # Process epoch metrics
            for key, value in metrics.items():
                str_value = str(value)
                
                # Map to structured TrainingMetrics fields
                if key in ['loss', 'avg_loss', 'train_loss']:
                    training_metrics["loss"] = str_value
                elif key in ['accuracy', 'train_accuracy']:
                    training_metrics["accuracy"] = str_value
                else:
                    # Everything else goes to generic metrics
                    generic_metrics[key] = str_value

            # Add checkpoint information if available
            if checkpoint_dir and os.path.exists(checkpoint_dir):
                try:
                    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint-') or f.startswith('epoch-')]
                    if checkpoints:
                        training_metrics["checkpoints_stored"] = len(checkpoints)
                        # Find latest checkpoint by highest number
                        def get_checkpoint_number(checkpoint_name):
                            try:
                                if 'checkpoint-' in checkpoint_name:
                                    return int(checkpoint_name.split('-')[1].split('.')[0])
                                elif 'epoch-' in checkpoint_name:
                                    return int(checkpoint_name.split('-')[1].split('.')[0])
                                else:
                                    return -1
                            except (IndexError, ValueError):
                                return -1
                        
                        latest_checkpoint_name = max(checkpoints, key=get_checkpoint_number)
                        latest_checkpoint = os.path.join(checkpoint_dir, latest_checkpoint_name)
                        training_metrics["latest_checkpoint_path"] = latest_checkpoint
                except (OSError, ValueError):
                    pass

            # Store metrics for status writing
            self.training_metrics = training_metrics
            self.generic_metrics = generic_metrics

            epoch_num = epoch + 1
            total_epochs = self.total_epochs
            message = f"Completed epoch {epoch_num}/{total_epochs}"
            self.write_status(message)

        def write_status(self, message: str = "Training in progress"):
            """Write current training status to file."""
            try:
                current_time = time.time()
                
                # Basic status data
                status_data = {
                    "message": message,
                    "timestamp": int(current_time),
                    "start_time": int(self.start_time),
                    "current_step": self.current_step,
                    "total_steps": self.total_steps,
                    "current_epoch": self.current_epoch,
                    "total_epochs": self.total_epochs,
                }
                
                # Calculate percentage if we have step info
                if self.total_steps > 0:
                    percentage = (self.current_step / self.total_steps) * 100
                    status_data["percentage_complete"] = f"{percentage:.2f}"
                    
                    # Calculate ETA if we have progress
                    if self.current_step > 0:
                        elapsed_time = current_time - self.start_time
                        time_per_step = elapsed_time / self.current_step
                        remaining_steps = self.total_steps - self.current_step
                        eta_seconds = int(remaining_steps * time_per_step)
                        status_data["estimated_time_remaining"] = eta_seconds
                
                # Add structured training metrics if any
                if hasattr(self, 'training_metrics') and self.training_metrics:
                    status_data["training_metrics"] = self.training_metrics
                
                # Add generic metrics if any
                if hasattr(self, 'generic_metrics') and self.generic_metrics:
                    status_data["metrics"] = self.generic_metrics

                # Write to file atomically
                temp_file = f"{self.status_file_path}.tmp"
                with open(temp_file, "w") as f:
                    json.dump(status_data, f, indent=2)
                os.rename(temp_file, self.status_file_path)

            except Exception as e:
                print(f"Failed to write progression status: {e}")

    # Define the PyTorch CNN model to be trained
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

    def setup_distributed():
        """Initialize distributed training using operator-injected PET environment variables"""
        # Use PET_* environment variables injected by the training operator
        node_rank = int(os.getenv('PET_NODE_RANK', '0'))
        num_nodes = int(os.getenv('PET_NNODES', '1'))
        nproc_per_node = int(os.getenv('PET_NPROC_PER_NODE', '1'))
        master_addr = os.getenv('PET_MASTER_ADDR', 'localhost')
        master_port = os.getenv('PET_MASTER_PORT', '29500')
        
        # Calculate standard PyTorch distributed variables
        local_rank = int(os.getenv('LOCAL_RANK', '0'))
        world_size = num_nodes * nproc_per_node
        global_rank = node_rank * nproc_per_node + local_rank
        
        # Set standard PyTorch environment variables for compatibility
        os.environ['RANK'] = str(global_rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        os.environ['LOCAL_RANK'] = str(local_rank)
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
        
        # Use NCCL if a GPU is available, otherwise use Gloo as communication backend.
        device, backend = ("cuda", "nccl") if torch.cuda.is_available() else ("cpu", "gloo")
        print(f"Using Device: {device}, Backend: {backend}")
        
        # Initialize distributed training if world_size > 1
        if world_size > 1:
            try:
                torch.distributed.init_process_group(
                    backend=backend,
                    rank=global_rank,
                    world_size=world_size
                )
                torch.distributed.barrier()
                print(
                    "Distributed Training for WORLD_SIZE: {}, RANK: {}, LOCAL_RANK: {}".format(
                        world_size, global_rank, local_rank
                    )
                )
            except Exception as e:
                print(f"Warning: Failed to initialize distributed training: {e}")
        else:
            print("Single node training - distributed not initialized")
        
        return local_rank, global_rank, world_size, device

    def train_fashion_mnist():
        # Setup distributed training
        local_rank, global_rank, world_size, device_type = setup_distributed()

        # Create the model and load it into the device.
        if device_type == "cuda" and torch.cuda.is_available():
            device = torch.device(f"{device_type}:{local_rank}")
        else:
            device = torch.device("cpu")
        
        # Create model and wrap with DDP only if distributed
        net = Net().to(device)
        if world_size > 1:
            model = nn.parallel.DistributedDataParallel(net)
        else:
            model = net
        optimizer = torch.optim.SGD(model.parameters(), lr=float(os.getenv('LEARNING_RATE', '0.1')), momentum=0.9)
        
        # Setup checkpointing
        checkpoint_dir = os.getenv('CHECKPOINT_DIR', '/workspace/checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Resume from checkpoint if available
        start_epoch = 1
        if world_size == 1 or dist.get_rank() == 0:
            # Find latest checkpoint
            checkpoints = glob.glob(os.path.join(checkpoint_dir, 'epoch-*.pth'))
            if checkpoints:
                def get_epoch_number(checkpoint_path):
                    try:
                        filename = os.path.basename(checkpoint_path)
                        return int(filename.split('-')[1].split('.')[0])
                    except (IndexError, ValueError):
                        return -1
                
                latest_checkpoint = max(checkpoints, key=get_epoch_number)
                print(f"Resuming from checkpoint: {latest_checkpoint}")
                
                try:
                    # Load checkpoint with device mapping for CPU/GPU compatibility
                    checkpoint = torch.load(latest_checkpoint, map_location=device)
                    if world_size > 1:
                        model.module.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    print(f"Resumed from epoch {checkpoint['epoch']}, starting epoch {start_epoch}")
                except Exception as e:
                    print(f"Failed to load checkpoint: {e}")
                    print("Starting training from scratch...")
                    start_epoch = 1
        
        # Broadcast start_epoch to all ranks (only if distributed)
        if world_size > 1:
            if torch.cuda.is_available():
                start_epoch_tensor = torch.tensor(start_epoch, device=device)
                dist.broadcast(start_epoch_tensor, src=0)
                start_epoch = start_epoch_tensor.item()
            else:
                # For CPU training, use a different approach
                start_epoch_list = [start_epoch] if dist.get_rank() == 0 else [None]
                dist.broadcast_object_list(start_epoch_list, src=0)
                start_epoch = start_epoch_list[0]

        # Download FashionMNIST dataset only on local_rank=0 process.
        if local_rank == 0:
            dataset = datasets.FashionMNIST(
                "./data",
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor()]),
            )
        if world_size > 1:
            dist.barrier()
        dataset = datasets.FashionMNIST(
            "./data",
            train=True,
            download=False,
            transform=transforms.Compose([transforms.ToTensor()]),
        )

        # Shard the dataset across workers (only if distributed).
        if world_size > 1:
            train_loader = DataLoader(
                dataset,
                batch_size=int(os.getenv('BATCH_SIZE', '100')),
                sampler=DistributedSampler(dataset)
            )
        else:
            train_loader = DataLoader(
                dataset,
                batch_size=int(os.getenv('BATCH_SIZE', '100')),
                shuffle=True
            )

        # Initialize progression tracker (only on rank 0)
        tracker = None
        if world_size == 1 or dist.get_rank() == 0:
            num_epochs = int(os.getenv('NUM_EPOCHS', '5'))
            steps_per_epoch = len(train_loader)
            
            # Calculate total epochs for entire training plan
            total_epochs_planned = start_epoch + num_epochs - 1
            
            tracker = ProgressionTracker(
                total_epochs=total_epochs_planned, 
                steps_per_epoch=steps_per_epoch,
                update_interval=int(os.getenv('PROGRESSION_UPDATE_INTERVAL', '10'))
            )
            
            # Initialize tracker with cumulative progress across entire training lifecycle
            if start_epoch > 1:
                # Set progress based on completed epochs (cumulative)
                completed_epochs = start_epoch - 1
                tracker.current_epoch = completed_epochs
                tracker.current_step = completed_epochs * steps_per_epoch
                tracker.write_status(f"Training resumed from epoch {start_epoch}")
            else:
                tracker.current_epoch = 0
                tracker.current_step = 0
                tracker.write_status("Training started")

        if world_size > 1:
            dist.barrier()
        
        # Training loop with progression tracking and checkpointing
        num_epochs = int(os.getenv('NUM_EPOCHS', '5'))
        for epoch in range(start_epoch, start_epoch + num_epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            # Iterate over mini-batches from the training set
            for batch_idx, (inputs, labels) in enumerate(train_loader):
                # Copy the data to the GPU device if available
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = F.nll_loss(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track metrics for epoch average
                epoch_loss += loss.item()
                num_batches += 1

                # Update progression (only on rank 0)
                if tracker and batch_idx % 10 == 0:
                    current_lr = optimizer.param_groups[0]["lr"]
                    
                    # Calculate samples per second
                    current_time = time.time()
                    elapsed_time = current_time - tracker.start_time
                    total_samples_processed = (epoch - start_epoch) * len(train_loader) * int(os.getenv('BATCH_SIZE', '100')) + batch_idx * int(os.getenv('BATCH_SIZE', '100'))
                    samples_per_second = total_samples_processed / elapsed_time if elapsed_time > 0 else 0
                    
                    # Calculate accuracy (simple approximation)
                    with torch.no_grad():
                        _, predicted = torch.max(outputs.data, 1)
                        correct = (predicted == labels).sum().item()
                        accuracy = correct / labels.size(0)
                    
                    # Use absolute epoch for cumulative progress
                    tracker.update_step(
                        epoch=epoch,
                        step=batch_idx,
                        loss=loss.item(),
                        learning_rate=current_lr,
                        checkpoint_dir=checkpoint_dir,
                        accuracy=accuracy,
                        world_size=dist.get_world_size(),
                        local_rank=local_rank,
                        train_samples_per_second=f"{samples_per_second:.2f}",
                        train_runtime=f"{elapsed_time:.1f}",
                        grad_norm=f"{torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0):.4f}"
                    )

                if batch_idx % 10 == 0 and (world_size == 1 or dist.get_rank() == 0):
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(inputs),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )

            # Save checkpoint at the end of each epoch (only on rank 0)
            if world_size == 1 or dist.get_rank() == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'epoch-{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss / num_batches if num_batches > 0 else 0.0,
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

            # Update epoch progression (only on rank 0)
            if tracker:
                avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                
                # Calculate epoch accuracy (simple approximation)
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in train_loader:
                        data, target = data.to(device), target.to(device)
                        outputs = model(data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                
                epoch_accuracy = correct / total if total > 0 else 0.0
                
                # Use absolute epoch for cumulative progress
                tracker.update_epoch(
                    epoch=epoch,
                    checkpoint_dir=checkpoint_dir,
                    avg_loss=avg_loss,
                    accuracy=epoch_accuracy,
                    total_batches=num_batches,
                    total_samples=total
                )
                
            if world_size > 1:
                dist.barrier()

        # Wait for the distributed training to complete
        if world_size > 1:
            dist.barrier()
        
        if world_size == 1 or dist.get_rank() == 0:
            print("Training is finished")
            if tracker:
                # Write final completion status
                tracker.current_step = tracker.total_steps  # Ensure 100% completion
                tracker.write_status("Training completed")
                
                # Buffer time to ensure controller captures 100% completion
                print("Waiting for progression status to be captured...")
                time.sleep(30)  # Buffer time to update the progression status to 100%

        # Finally clean up PyTorch distributed
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == "__main__":
    train_fashion_mnist()