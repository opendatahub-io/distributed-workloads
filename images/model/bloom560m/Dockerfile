FROM registry.access.redhat.com/ubi9:9.4 as builder
WORKDIR /
RUN dnf install -y git-lfs

RUN git lfs install
RUN git clone https://huggingface.co/bigscience/bloom-560m
RUN rm -r bloom-560m/onnx
RUN rm -r bloom-560m/.git

FROM registry.access.redhat.com/ubi9:9.4
WORKDIR /

COPY --from=builder /bloom-560m /models/bloom-560m

