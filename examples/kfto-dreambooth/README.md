# Finetuning Stable Diffusion model using DreamBooth with Kubeflow Training on OpenShift AI

This example shows how user can finetune Stable Diffusion model using DreamBooth technique.
The finetuning is performed on OpenShift environment using Kubeflow Training operator and OpenShift AI.

This example is based on HuggingFace DreamBooth Hackathon example - https://huggingface.co/learn/diffusion-course/en/hackathon/dreambooth


## Requirements

* An OpenShift cluster with OpenShift AI (RHOAI) 2.17+ installed:
  * The `dashboard`, `trainingoperator` and `workbenches` components enabled
* Sufficient worker nodes for your configuration(s) with NVIDIA GPUs (Ampere-based or newer recommended) or AMD GPUs (AMD Instinct MI300X or newer recommended)
* AWS S3 storage available


## Setup

* Access the OpenShift AI dashboard, for example from the top navigation bar menu:
![](./docs/01.png)
* Log in, then go to _Data Science Projects_ and create a project:
![](./docs/02.png)
* Once the project is created, click on _Create a workbench_:
![](./docs/03.png)
* Then create a workbench with the following settings:
    * Select the `PyTorch` (or the `ROCm-PyTorch`) notebook image:
    ![](./docs/04a.png)
    * Select the `Medium` container size and add an accelerator:
    ![](./docs/04b.png)
        > [!NOTE]
        > Adding an accelerator is only needed to test the fine-tuned model from within the workbench so you can spare an accelerator if needed.
    * Keep the default 20GB workbench storage, it is enough to run the inference from within the workbench:
    * Click on _Create connection_ to create a workbench connection to your AWS S3 bucket:
    ![](./docs/04c.png)
    * Select _S3 compatible object storage_:
    ![](./docs/04d.png)
    * Fill all the needed fields, also specify _Bucket_ value (it is used in the workbench), then confirm :
    ![](./docs/04e.png)
        > [!NOTE]
        > If you use different connection name than _workbench-aws_ then you need to adjust the _aws_connection_name_ propery in notebook to refer to this new name.
    * Review the configuration and click "Create workbench":
    ![](./docs/04f.png)
* From "Workbenches" page, click on _Open_ when the workbench you've just created becomes ready:
![](./docs/05.png)
* From the workbench, clone this repository, i.e., `https://github.com/opendatahub-io/distributed-workloads.git`
* Navigate to the `distributed-workloads/examples/kfto-dreambooth` directory and open the `dreambooth` notebook

You can now proceed with the instructions from the notebook. Enjoy!