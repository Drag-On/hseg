cmake_minimum_required(VERSION 3.5)
project(hseg)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

### BOOST ###
find_package(Boost 1.54 COMPONENTS filesystem system REQUIRED)

### OpenCV ###
find_package(OpenCV REQUIRED)

### gco ###
add_subdirectory(lib/gco-v3.0)

set(SOURCE_FILES main.cpp src/UnaryFile.cpp include/UnaryFile.h src/k-prototypes/Clusterer.cpp include/k-prototypes/Clusterer.h include/Image/Image.h)
add_executable(hseg ${SOURCE_FILES})
target_include_directories(hseg PUBLIC include ${GCO_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hseg gco ${OpenCV_LIBS})
