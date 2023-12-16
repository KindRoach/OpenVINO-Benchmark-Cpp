FROM ubuntu:22.04
LABEL authors="kindroach"

ENV DEBIAN_FRONTEND=noninteractive

# Install dev tools
RUN apt-get update && apt-get install -y apt-utils build-essential cmake gdb git python3 python-is-python3 python3-pip

# Install 3rd party lib
RUN apt-get install -y libopencv-dev

# Install OpenVINO by APT, refer to:
# https://docs.openvino.ai/2023.0/openvino_docs_install_guides_installing_openvino_apt.html
RUN apt-get install -y wget gnupg
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB
RUN echo "deb https://apt.repos.intel.com/openvino/2023 ubuntu22 main" | tee /etc/apt/sources.list.d/intel-openvino-2023.list
RUN apt-get update && apt-get install -y openvino

# OpenVINO Dev Toolkit
RUN pip install openvino-dev

# intel GPU driver
RUN wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg
RUN echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | tee /etc/apt/sources.list.d/intel-gpu-jammy.list
RUN apt-get update && apt-get install -y \
    intel-opencl-icd intel-level-zero-gpu level-zero \
    intel-media-va-driver-non-free libmfx1 libmfxgen1 libvpl2 \
    libegl-mesa0 libegl1-mesa libegl1-mesa-dev libgbm1 libgl1-mesa-dev libgl1-mesa-dri \
    libglapi-mesa libgles2-mesa-dev libglx-mesa0 libigdgmm12 libxatracker2 mesa-va-drivers \
    mesa-vdpau-drivers mesa-vulkan-drivers va-driver-all vainfo hwinfo clinfo
