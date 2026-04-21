FROM quay.io/modh/ray:2.52.1-py312-cpu

WORKDIR /app

# Install deps
RUN pip install --no-cache-dir \
    docling==2.74.0 \
    pandas pyarrow fsspec s3fs

# Run as non-root (numeric is fine; OpenShift may override anyway)
USER 1001

CMD ["bash"]

