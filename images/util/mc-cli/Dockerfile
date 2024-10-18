FROM registry.access.redhat.com/ubi9:latest
WORKDIR /app
USER 1001

RUN curl https://dl.min.io/client/mc/release/linux-amd64/mc -o /app/mc
RUN chmod +x /app/mc
ENV PATH="$PATH:/app"
ENV MC_CONFIG_DIR="/app"
RUN chown -R $USER:0 /app && \
    chmod -R g+rwX /app
