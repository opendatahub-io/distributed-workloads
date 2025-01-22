# Use the official Go image as a base image
FROM golang:1.21

ENV KUBECONFIG=/distributed-workloads/tests/.kube/config

# Set the working directory inside the container
WORKDIR /distributed-workloads

# Copy the go mod and sum files
COPY go.mod go.sum ./

# Download all dependencies
RUN go mod download && \
    go install gotest.tools/gotestsum@latest
    # go install github.com/jstemmer/go-junit-report/v2@latest

WORKDIR /distributed-workloads/tests

# Copy the source from the current directory to the working directory inside the container
COPY tests/ .

# Command to run the tests
ENTRYPOINT [ "gotestsum"]
