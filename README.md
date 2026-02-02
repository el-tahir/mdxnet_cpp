
# MDXNet C++

A C++ implementation that runs MDX-Net ONNX models (right now it supports `UVR_MDXNET_KARA_2.onnx`). The model separates vocals from instrumental tracks and returns the instrumental.

## Features

- **STFT/ISTFT Processing** – Custom DSP implementation using Kiss FFT
- **ONNX Runtime Inference** – Runs MDX-Net models for vocal separation
- **WAV File Support** – Reads and writes stereo WAV files

## Demo

### Before & After Audio Separation

**Original (with vocals):**



**After Processing (instrumental only):**



## Requirements

- C++ compiler with C++17 support
- CMake (version 3.18 or higher)
- `make` utility

## Building

The project features a fully automated build system. Running `make` from the root directory handles everything needed to get the application running.

To build the project:

```bash
make
```

This single command will:
1. **Setup ONNX Runtime:** Automatically download and extract the required libraries.
2. **Download Model:** Download the `UVR_MDXNET_KARA_2.onnx` model into the `models/` directory.
3. **Compile:** Build the `separator` executable inside the `build/` directory.

The final executable will be located at `build/separator`.

### Cleaning the build
To remove all build artifacts (excluding downloaded libraries/models):
```bash
make clean
```

## Usage

The application automatically handles audio preprocessing using `ffmpeg` (which must be installed on your system). It converts any input audio to the required 16-bit PCM 44.1kHz stereo format internally.

```bash
./build/separator <input_file> <output.wav>
```

**Example:**
```bash
./build/separator song.mp3 instrumental.wav
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
