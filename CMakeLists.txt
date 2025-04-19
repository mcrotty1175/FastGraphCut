cmake_minimum_required(VERSION 3.8)
project(FastGrabCut LANGUAGES CXX CUDA)

find_package(OpenMP)
if (OPENMP_FOUND)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(OpenCV_DIR "/afs/ece.cmu.edu/usr/$ENV{USER}/Private/build/")

find_package( OpenCV REQUIRED )
find_package( CUDA 12.3 REQUIRED )

set(CMAKE_CUDA_ARCHITECTURES 52 60 61 70 75 80 86)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

include_directories( ${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

link_directories(${CUDA_LIBRARY_DIRS})

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

add_executable(SlowGrabCut benchmark/grabcut.cpp )
target_link_libraries(SlowGrabCut
    ${OpenCV_LIBS}
)

cuda_add_executable(FastGrabCut gpu/grabcut.cu )
target_link_libraries(FastGrabCut
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${CUDA_npps_LIBRARY}
    ${CUDA_nppi_LIBRARY}
)
