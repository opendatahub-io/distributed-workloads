# Batch Document Processing with Ray and Docling on OpenShift AI

This example shows how to scale the CPU-intensive task of document conversion using Ray Data to parallelize Docling across multiple worker nodes. The notebook requires an image with Codeflare-sdk, Ray and Docling.

> [!IMPORTANT]
> This example has been tested with a sample dataset.
> The configuration space is highly dimensional, with application configuration tightly coupled to runtime / hardware configuration.
> It is your responsibility to adapt it, and validate it works as expected, with your configuration(s), on your target environment(s).

## Requirements

- OpenShift / RHOAI cluster requirements
- CodeFlare SDK installed in the workbench
- A ReadWriteMany PVC mounted on the RayCluster (head + workers)
  - Input PDFs in a subdirectory (default: /mnt/data/input/pdfs)
- An existing Ray cluster with Docling and dependencies installed

## Running the Example

1. Authenticate to OpenShift (oc login)
2. Clone the repo:  [https://github.com/opendatahub-io/distributed-workloads.git](https://github.com/opendatahub-io/distributed-workloads.git) and navigate to **examples/ray-docling**
3. Open **ray_data.ipynb**
4. Update configuration variables (cluster name, namespace, PVC, actor/CPU settings)
5. Run all cells — the notebook generates ray_data_process_async.py and submits it
6. Monitor with rayjob.status() and retrieve logs with oc logs

## Helpful Information

- You may need to tweak the number of CPUs available for each worker depending on the number of documents and their sizes.

