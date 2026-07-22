# Docker NVIDIA Notes
[Back to Index ../](../index.md)

## Setting Up Docker Containers for NVIDIA GPUs

To run Docker on NVIDIA GPUs, you need to install the NVIDIA Container Toolkit first.
Follow the instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

Once the NVIDIA Container Toolkit has been installed, you can check access to the GPU by running the following image to run the `nvidia-smi` tool:

```bash
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

You will get an output similar to this:

```bash
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 590.48.01              Driver Version: 590.48.01      CUDA Version: 13.1     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 5060        Off |   00000000:01:00.0  On |                  N/A |
|  0%   46C    P8             12W /  145W |     578MiB /   8151MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

## DockerFile for HAT using NVIDIA SDK

You can use the following `Dockerfile` for building a new docker container:

```dockerfile
FROM nvidia/cuda:13.0.1-devel-ubuntu24.04

RUN apt-get update -q && apt install -qy \
        build-essential git cmake vim maven curl bash unzip zip wget

WORKDIR /opt/babylon/
RUN wget https://download.java.net/java/early_access/jdk26/22/GPL/openjdk-26-ea+22_linux-x64_bin.tar.gz
RUN tar xvzf openjdk-26-ea+22_linux-x64_bin.tar.gz
ENV JAVA_HOME=/opt/babylon/jdk-26/
ENV PATH=$JAVA_HOME/bin:$PATH
RUN java --version

## Configure Babylon/HAT from source
RUN git clone https://github.com/openjdk/babylon.git
WORKDIR /opt/babylon/babylon

RUN apt-get update -y
RUN apt-get install -y autoconf libfreetype6-dev
RUN apt-get install -y file
RUN apt-get install -y libasound2-dev
RUN apt-get install -y libcups2-dev
RUN apt-get install -y libfontconfig1-dev
RUN apt-get install -y libx11-dev libxext-dev libxrender-dev libxrandr-dev libxtst-dev libxt-dev

RUN bash configure --with-boot-jdk=${JAVA_HOME}
RUN make clean
RUN make images

# Configure HAT
WORKDIR /opt/babylon/babylon/hat
RUN wget https://download.java.net/java/early_access/jextract/22/6/openjdk-22-jextract+6-47_linux-x64_bin.tar.gz
RUN tar xvzf openjdk-22-jextract+6-47_linux-x64_bin.tar.gz > /dev/null
ENV PATH=/opt/babylon/babylon/hat/jextract-22/bin:$PATH
ENV PATH=/opt/babylon/babylon/build/linux-x86_64-server-release/jdk/bin/:$PATH
ENV JAVA_HOME=/opt/babylon/babylon/build/linux-x86_64-server-release/jdk
RUN /bin/bash -c "source env.bash"

RUN apt-get install -y maven
RUN mvn clean package

## Expose a volume to pass files in the local directory
WORKDIR /opt/babylon/babylon/hat/
VOLUME ["/data"]
```

## Build Image

Run the following command in the same directory of the `Dockerfile` with the previous configuration:

```bash
docker build . -t babylon
```

## Running Examples on the NVIDIA GPU

Check `nvidia-smi` tool from NVIDIA with the new image, so we have connection to the GPU:

```bash
docker run -it --rm --runtime=nvidia --gpus all babylon nvidia-smi
```

All setup! Now you can run HAT on NVIDIA GPUs.

Run matrix-multiply example:

```bash
docker run -it --rm --runtime=nvidia --gpus all babylon java @.ffi-cuda-examples matmul --size=1024 --kernel=2DREGISTERTILING_FP16
```

## Enable debug info

```bash
docker run -it --rm --runtime=nvidia --gpus all babylon java @.ffi-cuda-examples matmul -DHAT=INFO matmul --size=1024 --kernel=2DREGISTERTILING_FP16
```

Expected output:

```bash
[INFO] Input Size     : 1024x1024
[INFO] Check Result:  : false
[INFO] Num Iterations : 100
[INFO] NDRangeConfiguration: 2DREGISTER_TILING_FP16

[INFO] Using NVIDIA GPU: NVIDIA GeForce RTX 5060
[INFO] Dispatching the CUDA kernel
        \_ BlocksPerGrid   = [16,16,1]
        \_ ThreadsPerBlock = [16,16,1]
```