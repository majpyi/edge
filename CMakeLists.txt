cmake_minimum_required(VERSION 3.10)
project(edge)

set(CMAKE_CXX_STANDARD 11)

#find_library(OpenCV)
find_package(OpenCV)

include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 11)

include_directories(header)

add_executable(edge
        header/basic_gx.h
        header/modeltolearn.h
        header/transform.h
        header/use.h
        basic_gx.cc
        make_dir.cc
        ps_way.cc
        transform_pic.cc
        modeltolearn.cc
        )
target_link_libraries(edge ${OpenCV_LIBS})