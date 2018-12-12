cmake_minimum_required(VERSION 2.8)

project(c4we CXX)
set(CMAKE_CXX_FLAGS "-std=c++14 -Wall -g -fPIC -O3 -march=native -ffast-math")
set(CMAKE_CXX_COMPILER "clang++")

find_package(Armadillo 7.0 REQUIRED)
find_package(yaml-cpp 0.5 REQUIRED)

if(ARMADILLO_FOUND)
  message("armadillo found")
  include_directories(${ARMADILLO_INCLUDE_DIRS})
else()
  message(FATAL_ERROR "Cannot find Armadillo")
endif()

if(yaml-cpp_FOUND)
  message("yaml-cpp found")
  include_directories(${YAML_CPP_INCLUDE_DIR})
else()
  message(FATAL_ERROR "Cannot find yaml-cpp")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

add_subdirectory(src)
