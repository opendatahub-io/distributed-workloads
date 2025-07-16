# Fine-Tuning a RAG Model with Feast on OpenShift AI

This project provides an end-to-end example of how to fine-tune a Retrieval-Augmented Generation (RAG) model on **OpenShift AI**. It uses the **Feast** (Feature Store) for efficient retrieval of context and the **Kubeflow Training SDK** to orchestrate the distributed fine-tuning job on the cluster.

The core idea is to enhance a generator model (like BART) by providing it with relevant documents retrieved from a knowledge base at runtime. This notebook handles the entire lifecycle: ingesting data into the feature store, fine-tuning the RAG model on synthetically generated Q&A pairs, and testing the final artifact.

***

## Prerequisites

Before you begin, ensure you have the following setup:

* An OpenShift cluster with OpenShift AI (RHOAI) 2.20+ installed:
  * The `dashboard`, `trainingoperator` and `workbenches` components enabled.
* Workbench with medium size container, 1 NVIDIA GPU / 1 AMD GPU accelerator, and cluster storage of 200GB.
* Sufficient worker nodes for your configuration(s) with NVIDIA GPUs (Ampere-based or newer recommended) or AMD GPUs depending on your environment.
* A dynamic storage provisioner supporting RWX PVC provisioning.
* A standalone Milvus deployment. See example [here](https://github.com/rh-aiservices-bu/llm-on-openshift/tree/main/vector-databases/milvus#deployment).

***

## Workbench Setup

You must run this notebook from within an OpenShift AI Workbench. Follow these steps to create one:

* Access the OpenShift AI dashboard:
* Log in, then go to _Data Science Projects_ and create a project.
* Once the project is created, click on _Create a workbench_.
* Then create a workbench with a preferred name and with the following settings:
  * Select the `PyTorch` (or the `ROCm-PyTorch`) workbench image with the recommended version.
  * Select the `Medium` as the deployment container size.
  * Add one NVIDIA / AMD accelerator (GPU) depending on environment.
  * Create a storage that'll be shared between the workbench and the fine-tuning runs.
    Make sure it uses a storage class with RWX capability and give it enough capacity according to the size of the model you want to fine-tune.
    > [!NOTE]
    > You can attach an existing shared storage if you already have one instead.
  * Review the storage configuration and click "Create workbench"
* From the "Workbenches" page, click the icon ![Open icon](https://raw.githubusercontent.com/primer/octicons/main/icons/link-external-16.svg) on the workbench you've just created once it becomes ready.
* From the workbench, clone this repository, i.e., `https://github.com/opendatahub-io/distributed-workloads.git`
* Navigate to the `distributed-workloads/examples/kfto-sft-feast-rag` directory and open the `sft_feast_rag_model` notebook

You can now proceed with the instructions from the notebook.
***

## Workflow Overview

The notebook is structured to guide you through the following key stages:

* **feature_repo/feature_store.yaml**
  This is the core configuration file for the RAG project's feature store, configuring a Milvus online store on a local provider.
  * In order to configure Milvus you should:
    - Update `feature_store.yaml` with your Milvus connection details:
      - host
      - port (default: 19530)
      - credentials (if required)
* **Environment Setup**: Installs all necessary Python libraries, including the Kubeflow Training SDK, Feast, and Hugging Face Transformers.
* **Data Ingestion with Feast**:
  * Loads the `facebook/wiki_dpr` dataset.
  * Chunks the documents into smaller, manageable passages.
  * Generates vector embeddings for each passage using a `sentence-transformers` model.
  * Initializes a Feast feature store and ingests the passages and their embeddings into the online store.
* **Distributed Training**:
  * The core training logic is defined in a `main` function.
  * It defines a `RagSequenceForGeneration` model, combining a question-encoder with a generator model.
  * It uses a custom `FeastRAGRetriever` to connect the RAG model to the Feast feature store.
  * The notebook uses the Kubeflow `TrainingClient` to submit this `main` function as a distributed `PyTorchJob` to the OpenShift cluster.
* **Monitoring**: You can monitor the job's progress directly through its logs and visualize metrics using the integrated TensorBoard dashboard.
* **Inference and Testing**: After the training job is complete, the final, fine-tuned RAG model is loaded from shared storage for testing.

***
