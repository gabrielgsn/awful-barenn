# Ambiente único para build, benchmark e callgrind (CPU) e nvcc (CUDA).
#   docker build -t barenn-env docker
#   docker run --rm --gpus all -v "$PWD":/work -w /work barenn-env make bench
FROM nvidia/cuda:12.6.3-devel-ubuntu22.04
RUN apt-get update && apt-get install -y --no-install-recommends valgrind make git \
    && rm -rf /var/lib/apt/lists/*
