#prueba 1
# Base image for GitHub Actions workflow
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Instalamos únicamente las dependencias y compiladores
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     ca-certificates \
     cmake \
     g++ \
     make \
     git \
     libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["bash"]