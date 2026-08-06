# DwarfStar 4 CUDA container

## Storage prerequisite

The atomic image embeds the 86.7 GB GGUF. Keep at least 180 GiB free on the
physical disk used by Docker Desktop before building it: Docker temporarily
stores the build context, the image layer, and its unpacked snapshot. The
finished image needs no model bind mount, but it still requires the host
NVIDIA driver and NVIDIA Container Toolkit to expose the GPU.

Build the atomic Blackwell image from the repository root. The named `model`
context adds the GGUF as a separate immutable image layer:

```sh
docker build \
  --build-context model=/home/peppe200175/ds4/gguf \
  -f Dockerfile.cuda \
  -t ds4-cuda:0731-atomic .
```

The resulting image contains the CUDA runtime, all DwarfStar executables,
profiles, launcher, and the 81 GB DeepSeek V4 Flash 0731 GA GGUF. No host
volume is required. Start an interactive shell with NVIDIA GPU access:

```sh
docker run --rm -it --gpus all ds4-cuda:0731-atomic
```

If Docker Desktop cannot ingest the 86.7 GB named context in one BuildKit
operation, the equivalent atomic image can be assembled through a staging
container. This path has the same physical free-space requirement:

```sh
docker build --target runtime-base -f Dockerfile.cuda -t ds4-cuda:0731-runtime .
docker create --name ds4-atomic-staging ds4-cuda:0731-runtime
docker cp /home/peppe200175/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf \
  ds4-atomic-staging:/models/
docker commit ds4-atomic-staging ds4-cuda:0731-atomic
docker rm ds4-atomic-staging
```

Inside the container, start the optimized interactive CLI with:

```sh
ds4-run
```

For a one-shot smoke test:

```sh
ds4-run -p "Reply only OK" -n 1 --temp 0
```

The image defaults match the measured RTX 5080 configuration: a 10 GB routed
expert budget, 128-token prefill chunks, layer-local decode LRU, four persistent
direct-I/O readers, and 256 MiB CUDA weight-arena chunks. Layer-pinned profiles
remain opt-in; for example:

```sh
DS4_CUDA_PINNED_EXPERTS_FILE=/opt/ds4/profiles/ds4-flash-0731-prefill-pinned-conservative.txt ds4-run
```
