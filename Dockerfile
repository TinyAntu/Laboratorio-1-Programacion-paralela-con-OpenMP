# Base image for GitHub Actions workflow
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Instalamos dependencias, compiladores y utilidades de Python para graficar
RUN apt-get update \
  && apt-get install -y --no-install-recommends \
     ca-certificates \
     cmake \
     g++ \
     make \
     git \
     libgomp1 \
     #para graficar con python
     python3 \
     python3-pandas \
     python3-matplotlib \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["bash"]