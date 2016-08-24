cmake_minimum_required(VERSION 3.5)
project(hseg)

### Full optimization ###
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fprofile-use=.profiling")


### BOOST ###
find_package(Boost 1.54 COMPONENTS filesystem system REQUIRED)

### OpenCV ###
find_package(OpenCV REQUIRED)

### gco ###
add_subdirectory(lib/gco-v3.0)

### properties ###
add_subdirectory(lib/properties)

### Set general gcc flags ###
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -Wpedantic")

### Function to prepend a string to all elements of a list
FUNCTION(PREPEND var prefix)
    SET(listVar "")
    FOREACH (f ${ARGN})
        LIST(APPEND listVar "${prefix}/${f}")
    ENDFOREACH (f)
    SET(${var} "${listVar}" PARENT_SCOPE)
ENDFUNCTION(PREPEND)

set(SOURCE_FILES main.cpp src/Energy/UnaryFile.cpp include/Energy/UnaryFile.h src/k-prototypes/Clusterer.cpp include/k-prototypes/Clusterer.h include/Image/Image.h include/k-prototypes/Cluster.h include/helper/coordinate_helper.h include/k-prototypes/Feature.h src/k-prototypes/Feature.cpp src/k-prototypes/Cluster.cpp include/helper/image_helper.h src/helper/image_helper.cpp include/helper/opencv_helper.h src/helper/opencv_helper.cpp src/GraphOptimizer/GraphOptimizer.cpp include/GraphOptimizer/GraphOptimizer.h src/Energy/EnergyFunction.cpp include/Energy/EnergyFunction.h include/Properties.h include/Image/Coordinates.h src/Energy/Weights.cpp include/Energy/Weights.h include/helper/hash_helper.h)
set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include ${gco_INCLUDE_DIRS} ${properties_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
set(LIBS gco properties ${OpenCV_LIBS})
PREPEND(SOURCE_FILES_FULL_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCE_FILES})

add_executable(hseg ${SOURCE_FILES})
target_include_directories(hseg PUBLIC ${INCLUDE_DIRS})
target_link_libraries(hseg ${LIBS})

### Use any sanitizer? ###
add_subdirectory(sanitizer/address)
add_subdirectory(sanitizer/leak)
add_subdirectory(sanitizer/thread)
add_subdirectory(sanitizer/undefined)
add_subdirectory(sanitizer/gprof)
add_subdirectory(sanitizer/profile)