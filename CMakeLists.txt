cmake_minimum_required(VERSION 3.14)
project(mdxnet_cpp VERSION 1.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type (default to Release for optimized builds)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_C_FLAGS_RELEASE "-O2")

# ONNX Runtime paths
set(ONNXRUNTIME_ROOT "${CMAKE_SOURCE_DIR}/onnxruntime")
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

# Check if ONNX Runtime exists
if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}")
    message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}. "
            "Please download and extract it to the project directory.")
endif()

# Source files
set(SOURCES
    src/main.cpp
    src/DSPCore.cpp
    src/ModelHandler.cpp
    third_party/kiss_fft/kiss_fft.c
)

# Header files (for IDEs)
set(HEADERS
    include/DSPCore.h
    include/ModelHandler.h
    include/WAVHeader.h
    third_party/kiss_fft/kiss_fft.h
    third_party/kiss_fft/_kiss_fft_guts.h
    third_party/kiss_fft/kiss_fft_log.h
)

# Main executable
add_executable(separator ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(separator PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party/kiss_fft
    ${ONNXRUNTIME_INCLUDE_DIR}
)

# Link directories
target_link_directories(separator PRIVATE
    ${ONNXRUNTIME_LIB_DIR}
)

# Link libraries
target_link_libraries(separator PRIVATE
    onnxruntime
)

# Set RPATH for finding shared libraries at runtime
set_target_properties(separator PROPERTIES
    BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
    INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
)

# Optional: Build test executables
option(BUILD_TESTS "Build test executables" OFF)

if(BUILD_TESTS)
    # DSP test
    add_executable(audio_test 
        tests/test_dsp.cpp 
        src/DSPCore.cpp 
        third_party/kiss_fft/kiss_fft.c
    )
    target_include_directories(audio_test PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/third_party/kiss_fft
    )
    
    # Model test
    add_executable(model_test 
        tests/test_model_handler.cpp 
        src/ModelHandler.cpp
    )
    target_include_directories(model_test PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${ONNXRUNTIME_INCLUDE_DIR}
    )
    target_link_directories(model_test PRIVATE ${ONNXRUNTIME_LIB_DIR})
    target_link_libraries(model_test PRIVATE onnxruntime)
    set_target_properties(model_test PROPERTIES
        BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}"
        INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}"
    )
endif()

# Print build info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
