cmake_minimum_required (VERSION 2.6)
project(densecrf)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)

set(dense_crf_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include CACHE PATH "Include directory for the denseCRF library")
include_directories( ${dense_crf_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} )

add_subdirectory (src)
add_subdirectory (examples)
add_subdirectory (external)
