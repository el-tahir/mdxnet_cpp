
# MDXNet C++

A C++ implementation that runs MDX-Net ONNX models (right now it supports `UVR_MDXNET_KARA_2.onnx`). The model separates vocals from instrumental tracks and returns the instrumental.

## Features

- **STFT/ISTFT Processing** – Custom DSP implementation using Kiss FFT
- **ONNX Runtime Inference** – Runs MDX-Net models for vocal separation
- **WAV File Support** – Reads and writes stereo WAV files

## Requirements

- C++ compiler with C++11 support
- ONNX Runtime library
- An MDX-Net ONNX model (e.g., `UVR_MDXNET_KARA_2.onnx`)

## Installing ONNX Runtime

1. Download the ONNX Runtime release for your platform from the [official releases page](https://github.com/microsoft/onnxruntime/releases).

2. Extract and place it in the project directory:

```bash
# Example for Linux x64
wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz
tar -xzf onnxruntime-linux-x64-1.16.3.tgz
mv onnxruntime-linux-x64-1.16.3 onnxruntime
```

3. Your directory structure should look like:

```
mdxnet_cpp/
├── onnxruntime/
│   ├── include/
│   └── lib/
├── main.cpp
└── ...
```

## Downloading the Model

  

Download the `UVR_MDXNET_KARA_2.onnx` model from Hugging Face:

  

```bash

wget  https://huggingface.co/AI4future/RVC/resolve/main/UVR_MDXNET_KARA_2.onnx

```

  

Or download it manually from the [Hugging Face page](https://huggingface.co/AI4future/RVC/blob/main/UVR_MDXNET_KARA_2.onnx).

  

Place the model file in the project root directory.

## Building

```bash
g++ -O2 -I./onnxruntime/include -L./onnxruntime/lib \
-Wl,-rpath,./onnxruntime/lib \
-o separator main.cpp DSPCore.cpp ModelHandler.cpp kiss_fft.c \
-lonnxruntime
```

## Usage

```bash
./separator <input.wav> <output.wav>
```

**Example:**

```bash
./separator song.wav instrumental.wav
```

## Project Structure

| File | Description |
|------|-------------|
| `main.cpp` | Entry point and separation pipeline |
| `DSPCore.cpp/h` | STFT/ISTFT and audio processing |
| `ModelHandler.cpp/h` | ONNX model loading and inference |
| `WAVHeader.h` | WAV file I/O utilities |
| `utils.cpp` | Tensor conversion helpers |
| `kiss_fft.c/h` | FFT library |
```
