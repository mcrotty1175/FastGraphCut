cmake_minimum_required(VERSION 2.8.12.2)
project(FastGrabCut)

set(OpenCV_DIR "/afs/ece.cmu.edu/usr/abatrach/Private/build/")

find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS})

find_package(OpenMP)
if (OPENMP_FOUND)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


add_executable(FastGrabCut benchmark/grabcut.cpp )
target_link_libraries(FastGrabCut ${OpenCV_LIBS})