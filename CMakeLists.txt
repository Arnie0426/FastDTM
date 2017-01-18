project(FastDTM)
include_directories("/usr/local/Cellar/eigen/3.2.8")
cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp dtm.cpp alias_samples.cpp)

add_executable(FastDTM ${SOURCE_FILES})