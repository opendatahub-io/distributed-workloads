# Machine Learning Model Optimization and Deployment with Ray and CodeFlare SDK

This code primarily focuses on optimizing machine learning models using RayTune. It demonstrates the process of hyperparameter tuning to find the best-performing model configuration for a given model, leveraging the capabilities of RayCluster and CodeFlare."


## Prerequisites

1. **Create an OpenShift AI Project:** Before starting, ensure you have an OpenShift AI account. Follow the steps outlined in the [OpenShift AI tutorial](https://access.redhat.com/documentation/en-us/red_hat_openshift_ai_cloud_service/1/html/openshift_ai_tutorial_-_fraud_detection_example/setting-up-a-project-and-storage#doc-wrapper) to set up your project and storage.
2. **Set Up Workbench:** Once your project is created, set up a workbench using the Standard Data Science image provided. This will serve as your development platform for building and optimizing machine learning models.
3. **Create Data Connections:** Establish connections to external data sources, such as an S3 bucket, where you'll upload your trained models.


## Code Overview

1. **Installation**: The code starts by installing the necessary dependencies using `pip`.
2. **Authentication**: It sets up authentication for accessing the cluster. Replace the `TOKEN` and `SERVER` values with actual authentication tokens and server URLs.
3. **Cluster Configuration**: Configures the Ray cluster with the desired parameters such as number of workers, CPU and memory allocation.
4. **Cluster Deployment**: Brings up the Ray cluster using the configuration defined.
5. **Hyperparameter Tuning & log Metadata**: Uses Ray Tune to perform hyperparameter tuning on a simple neural network model while concurrently logging metadata.
6. **Model Deployment**: Saves the best model obtained from hyperparameter tuning in the ONNX format and uploads it to an S3 bucket.
7. **REST API Integration**: Sets up details to access the deployed model through a REST API.

## Logging Metadata

Logging metadata is crucial for tracking the lineage, provenance, and performance of machine learning models. Use the provided functions or methods to log metadata during different stages of your workflow:

- Define Metadata Types: Define the metadata types for artifacts and contexts. Artifacts include HPOConfig, DataSet, Metrics, Model; Contexts inclue HPOTrial, HPOExperiment and HPOTrainer as ExecutionType using metadata store service.
- Create Parent Context: Create a parent context to associate trial contexts with the experiment context.
- Log Inputs and Outputs: Log input artifacts(DataSet, HPOCOnfig), events, and executions(HPOTrainer) when training and evaluating models.
- Log output artifacts, such as trained models and evaluation metrics, after model training and evaluation.
- Retrieve Artifacts and Contexts: Retrieve artifacts and contexts from the model registry based on specific conditions or trial names.


## Working with Model Registry

### Setting Up Model Registry

To install the Model Registry Controller, follow these steps:

1. Clone and navigate to the Model Registry Operator repository:

    ```bash
    git clone https://github.com/opendatahub-io/model-registry-operator.git
    cd model-registry-operator
    ```
2. Use the provided Makefile to deploy the operator. Specify the image location using the `IMG` argument. For example, to deploy from a latest image hosted on Quay.io, run:

    ```bash
    make deploy IMG=quay.io/opendatahub/model-registry-operator
    ```

This command will deploy the Model Registry Controller using the specified image.

### Starting the Model Registry Service

After deploying the Model Registry Controller, you can start the Model Registry service with either PostgreSQL or MySQL as the backend database.
To start the Model Registry service with PostgreSQL, use the following command:

```bash
kubectl apply -k config/samples/postgres
```

## Note

- Ensure you have access to the specified S3 bucket and have the necessary permissions to upload the files.
- Adjust the hyperparameter search space and training loop as needed for your specific use case.
- Make sure to replace placeholder values such as `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_S3_ENDPOINT`, `AWS_DEFAULT_REGION`, and `AWS_S3_BUCKET` with your actual AWS credentials and bucket details.

By following these instructions, you can effectively leverage the provided code to discover the optimal model using RayTune and subsequently deploy and serve machine learning models using Ray and the CodeFlare SDK.


