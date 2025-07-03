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

# Copy the .env-odh file and the run-test.sh script to the working directory inside the container
COPY images/tests/.env-odh .env-odh
COPY images/tests/run-test.sh .

# Command to run the tests
ENTRYPOINT ["bash", "run-test.sh"]
