cmake_minimum_required(VERSION 3.5)
project(hseg)

### BOOST ###
find_package(Boost 1.54 COMPONENTS filesystem system REQUIRED)

### OpenCV ###
find_package(OpenCV REQUIRED)

### gco ###
add_subdirectory(lib/gco-v3.0)

### properties ###
add_subdirectory(lib/properties)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Wpedantic")

set(SOURCE_FILES main.cpp src/Energy/UnaryFile.cpp include/Energy/UnaryFile.h src/k-prototypes/Clusterer.cpp include/k-prototypes/Clusterer.h include/Image/Image.h include/k-prototypes/Cluster.h include/helper/coordinate_helper.h include/k-prototypes/Feature.h src/k-prototypes/Feature.cpp src/k-prototypes/Cluster.cpp include/helper/image_helper.h src/helper/image_helper.cpp include/helper/opencv_helper.h src/helper/opencv_helper.cpp src/GraphOptimizer/GraphOptimizer.cpp include/GraphOptimizer/GraphOptimizer.h src/Energy/EnergyFunction.cpp include/Energy/EnergyFunction.h include/Properties.h include/Image/Coordinates.h src/Energy/Weights.cpp include/Energy/Weights.h)
add_executable(hseg ${SOURCE_FILES})
target_include_directories(hseg PUBLIC include ${gco_INCLUDE_DIRS} ${properties_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(hseg gco properties ${OpenCV_LIBS})
