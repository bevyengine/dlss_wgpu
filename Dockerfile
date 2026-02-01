FROM rust:latest

RUN git clone -b v310.5.0 https://github.com/NVIDIA/DLSS.git
RUN apt-get update && apt-get install --no-install-recommends -yq libclang-dev
RUN curl -O https://sdk.lunarg.com/sdk/download/1.4.335.0/linux/vulkansdk-linux-x86_64-1.4.335.0.tar.xz && tar --xz -x -f vulkansdk-linux-x86_64-1.4.335.0.tar.xz
RUN rustup component add rustfmt clippy

ENV DLSS_SDK=/DLSS
ENV VULKAN_SDK=/1.4.335.0/x86_64/

WORKDIR /src
