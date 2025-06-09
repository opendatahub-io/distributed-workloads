# Use the official Go image as a base image
FROM golang:1.23

ENV KUBECONFIG=/distributed-workloads/tests/.kube/config

# Set the working directory inside the container
WORKDIR /distributed-workloads

# Download the latest OpenShift CLI binary
RUN wget -q https://mirror.openshift.com/pub/openshift-v4/clients/ocp/stable-4.18/openshift-client-linux.tar.gz -P oc-client && \
    tar -xf oc-client/openshift-client-linux.tar.gz -C oc-client && \
    cp oc-client/oc /usr/local/bin && \
    rm -rf oc-client/

# Copy the go mod and sum files
COPY go.mod go.sum ./

# Download all dependencies
RUN go mod download && \
    go install gotest.tools/gotestsum@latest

WORKDIR /distributed-workloads/tests

# Copy the source from the current directory to the working directory inside the container
COPY tests/ .

# Command to run the tests
ENTRYPOINT [ "gotestsum"]

# Configure images using environment variables
ENV FMS_HF_TUNING_IMAGE=<replace_me>
ENV TEST_TRAINING_CUDA_PYTORCH_241_IMAGE=<replace_me>
ENV TEST_TRAINING_ROCM_PYTORCH_241_IMAGE=<replace_me>
ENV TEST_TRAINING_CUDA_PYTORCH_251_IMAGE=<replace_me>
ENV TEST_TRAINING_ROCM_PYTORCH_251_IMAGE=<replace_me>
ENV NOTEBOOK_IMAGE=<replace_me>
