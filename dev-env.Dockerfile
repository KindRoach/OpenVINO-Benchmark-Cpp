FROM continuumio/miniconda3
LABEL authors="kindroach"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

# Install dev tools
RUN apt-get install -y build-essential cmake gdb linux-tools-$(uname -r)

# Install 3rd party lib
RUN apt-get install -y libopencv-dev

# Install OpenVINO by APT, refer to:
# https://docs.openvino.ai/2023.0/openvino_docs_install_guides_installing_openvino_apt.html
RUN apt-get install -y wget gnupg
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
RUN echo "deb https://apt.repos.intel.com/openvino/2023 ubuntu22 main" | tee /etc/apt/sources.list.d/intel-openvino-2023.list
RUN apt-get update && apt-get install -y openvino
