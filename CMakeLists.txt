cmake_minimum_required(VERSION 3.6)
project(hseg)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

### BOOST ###
find_package(Boost 1.54 COMPONENTS filesystem system REQUIRED)

### gco ###
add_subdirectory(lib/gco-v3.0)

set(SOURCE_FILES main.cpp)
add_executable(hseg ${SOURCE_FILES})
target_include_directories(hseg PUBLIC include ${GCO_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
target_link_libraries(hseg gco)
