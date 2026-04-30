FROM quay.io/modh/ray:2.54.1-py312-cu128

USER 0
WORKDIR /app

RUN yum install -y mesa-libGL && yum clean all

ENV PIP_DEFAULT_TIMEOUT=600 PIP_RETRIES=10

RUN pip install --no-cache-dir \
    docling==2.74.0 \
    "vllm==0.10.2" \
    "pymilvus>=2.4.0" \
    pandas pyarrow fsspec s3fs boto3 \
    opencv-python-headless pypdfium2

# vLLM downgrades otel packages; restore base image versions so Ray dashboard works
RUN pip install --no-cache-dir --force-reinstall \
    "opentelemetry-sdk==1.41.0" \
    "opentelemetry-api==1.41.0" \
    "opentelemetry-semantic-conventions==0.62b0" \
    "opentelemetry-exporter-prometheus==0.62b0"

RUN pip uninstall -y uvloop

USER 1001
CMD ["bash"]
