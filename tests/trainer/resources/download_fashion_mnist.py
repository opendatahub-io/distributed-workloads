import os, gzip, shutil
from minio import Minio
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor

def main(dataset_path):
    # Download and Load Fashion-MNIST dataset 
    if all(var in os.environ for var in ["AWS_DEFAULT_ENDPOINT","AWS_ACCESS_KEY_ID","AWS_SECRET_ACCESS_KEY","AWS_STORAGE_BUCKET","AWS_STORAGE_BUCKET_FASHION_MNIST_DIR"]):
        print("Using provided storage bucket to download Fashion-MNIST datasets...")
        dataset_dir = os.path.join(dataset_path, "FashionMNIST/raw")
        endpoint = os.environ.get("AWS_DEFAULT_ENDPOINT")
        access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        bucket_name = os.environ.get("AWS_STORAGE_BUCKET")
        print(f"Storage bucket endpoint: {endpoint}")
        print(f"Storage bucket name: {bucket_name}\n")

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
        prefix=os.environ.get("AWS_STORAGE_BUCKET_FASHION_MNIST_DIR")
        print(f"Storage bucket Fashion-MNIST directory prefix: {prefix}\n")

        # download all files from prefix folder of storage bucket recursively
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
                ## Sample zipfilepath : ../data/FashionMNIST/raw/t10k-images-idx3-ubyte.gz
                with gzip.open(dataset_file_path, "rb") as f_in:
                    filename=file_name.split(".")[0]    #-> t10k-images-idx3-ubyte
                    file_path=("/".join(dataset_file_path.split("/")[:-1]))     #->../data/FashionMNIST/raw
                    full_file_path=os.path.join(file_path,filename)     #->../data/FashionMNIST/raw/t10k-images-idx3-ubyte
                    print(f"Extracting {dataset_file_path} to {file_path}..")

                    with open(full_file_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    print(f"Dataset file downloaded : {full_file_path}\n")
                # delete zip file
                os.remove(dataset_file_path)
            else:
                print(f"File-path '{dataset_file_path}' already exists")
        download_datasets = False
    else:
        print("Using default Fashion-MNIST mirror references to download datasets ...")
        print("Skipped usage of S3 storage bucket, because required environment variables aren't provided!\nRequired environment variables : AWS_DEFAULT_ENDPOINT, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_STORAGE_BUCKET, AWS_STORAGE_BUCKET_FASHION_MNIST_DIR")
        download_datasets = True

    datasets.FashionMNIST(
        dataset_path, 
        train=True, 
        download=download_datasets, 
        transform=Compose([ToTensor()])
        )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fashion-MNIST dataset download")
    parser.add_argument('--dataset_path', type=str, default="./data", help='Path to Fashion-MNIST datasets (default: ./data)')

    args = parser.parse_args()

    main(
        dataset_path=args.dataset_path,
    )

