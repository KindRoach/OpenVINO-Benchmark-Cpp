# OpenVINO-Benchmark-Cpp

### Setup Env

Conda is highly recommend to install third-party libraries.

```bash
conda create --name OpenVINO-Benchmark-Cpp
conda activate OpenVINO-Benchmark-Cpp
conda install -c conda-forge libopencv openvino
```

### Build

```bash
cmake -S . -B build
cmake --build build
```

> **_NOTE:_** if you are using CLion, please add `-DCMAKE_PREFIX_PATH=/path/to/conda/envs/OpenVINO-Benchmark-Cpp` as CMake options.