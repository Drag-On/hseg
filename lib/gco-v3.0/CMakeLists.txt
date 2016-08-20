cmake_minimum_required(VERSION 3.5)
project(gco)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(gco_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR} CACHE PATH "Include directory for the gco library")
set(GCO_SOURCE block.h energy.h GCoptimization.h graph.h LinkedBlockList.h GCoptimization.cpp graph.cpp LinkedBlockList.cpp maxflow.cpp)
add_library(gco ${gco_INCLUDE_DIRS} ${GCO_SOURCE})
