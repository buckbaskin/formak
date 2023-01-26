# syntax=docker/dockerfile:1
FROM ubuntu:focal-20221130
CMD ["bash"]
# required dependencies
RUN apt-get update  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ca-certificates curl autoconf
# Download and make bazelisk/bazel available
RUN curl -fsSL "https://github.com/bazelbuild/bazelisk/releases/download/v1.15.0/bazelisk-linux-amd64" > /tmp/bazelisk && mv /tmp/bazelisk /usr/local/bin/bazel
RUN ["chmod", "+x", "/usr/local/bin/bazel"]
# Set up a non-root user
RUN ["adduser", "--debug", "worker"]
USER worker
COPY --chown=worker:worker . .
# Warm up bazel
RUN ["bazel", "--version"]
CMD ["/usr/bin/bash"]