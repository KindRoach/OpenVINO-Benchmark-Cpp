# OpenVINO-Benchmark-Cpp

### Setup Env

#### OpenVINO

- Download from [link](https://storage.openvinotoolkit.org/repositories/openvino/packages/2023.2/windows/w_openvino_toolkit_windows_2023.2.0.13089.cfd42bd2cb0_x86_64.zip).
- Unzip to path ```lib/openvino```.

#### OpenCV

- Download from [link](https://github.com/opencv/opencv/releases/download/4.8.1/opencv-4.8.1-windows.exe).
- Install to path ```lib/opencv```.

### Build

```bash
cmake -S . -B build
cmake --build build
```

> **_NOTE:_** if you are using CLion, please add `-DCMAKE_PREFIX_PATH=/path/to/conda/envs/OpenVINO-Benchmark-Cpp` as CMake options.