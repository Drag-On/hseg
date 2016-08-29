cmake_minimum_required(VERSION 2.8)
project(hseg)

### Full optimization ###
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3") # -fprofile-use=.profiling

### Using any of the advanced compiler features? ###
option(ASAN "Enable gcc address sanitizer" OFF)
if (ASAN)
    set (CMAKE_CXX_FLAGS "-fsanitize=address")
endif()
option(LSAN "Enable gcc leak sanitizer" OFF)
if (LSAN)
    set (CMAKE_CXX_FLAGS "-fsanitize=leak")
endif()
option(TSAN "Enable gcc thread sanitizer" OFF)
if (TSAN)
    set (CMAKE_CXX_FLAGS "-fsanitize=thread")
endif()
option(USAN "Enable gcc undefined sanitizer" OFF)
if (USAN)
    set (CMAKE_CXX_FLAGS "-fsanitize=undefined")
endif()


### BOOST ###
find_package(Boost 1.58 COMPONENTS filesystem system REQUIRED)

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

set(SOURCE_FILES src/Energy/UnaryFile.cpp include/Energy/UnaryFile.h src/Inference/k-prototypes/Clusterer.cpp include/Inference/k-prototypes/Clusterer.h include/Image/Image.h include/Inference/k-prototypes/Cluster.h include/helper/coordinate_helper.h include/Inference/k-prototypes/Feature.h src/Inference/k-prototypes/Feature.cpp src/Inference/k-prototypes/Cluster.cpp include/helper/image_helper.h src/helper/image_helper.cpp include/helper/opencv_helper.h src/helper/opencv_helper.cpp src/Inference/GraphOptimizer/GraphOptimizer.cpp include/Inference/GraphOptimizer/GraphOptimizer.h src/Energy/EnergyFunction.cpp include/Energy/EnergyFunction.h include/Image/Coordinates.h src/Energy/Weights.cpp include/Energy/Weights.h include/helper/hash_helper.h src/Timer.cpp include/Timer.h src/Accuracy/ConfusionMatrix.cpp include/Accuracy/ConfusionMatrix.h src/Inference/InferenceIterator.cpp include/Inference/InferenceIterator.h include/Inference/InferenceResult.h include/Inference/InferenceResultDetails.h)
set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include ${gco_INCLUDE_DIRS} ${properties_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
set(LIBS gco properties ${OpenCV_LIBS} ${Boost_LIBRARIES})
PREPEND(SOURCE_FILES_FULL_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCE_FILES})

### Compile main library ###
add_library(hseg ${SOURCE_FILES})
target_include_directories(hseg PUBLIC ${INCLUDE_DIRS})
target_link_libraries(hseg ${LIBS})

### Compile clustering executable ###
add_executable(hseg_cluster exec/clustering.cpp)
target_include_directories(hseg_cluster PUBLIC ${INCLUDE_DIRS})
target_link_libraries(hseg_cluster ${LIBS} hseg)

### Compile inference executable ###
add_executable(hseg_infer exec/inference.cpp)
target_include_directories(hseg_infer PUBLIC ${INCLUDE_DIRS})
target_link_libraries(hseg_infer ${LIBS} hseg)

### Compile training executable ###
add_executable(hseg_train exec/train.cpp)
target_include_directories(hseg_train PUBLIC ${INCLUDE_DIRS})
target_link_libraries(hseg_train ${LIBS} hseg)

### Compile messy executable (i.e. the one where I do all my testing) ###
add_executable(hseg_mess exec/messy.cpp)
target_include_directories(hseg_mess PUBLIC ${INCLUDE_DIRS})
target_link_libraries(hseg_mess ${LIBS} hseg)

### Use any sanitizer? ###
#add_subdirectory(sanitizer/address)
#add_subdirectory(sanitizer/leak)
#add_subdirectory(sanitizer/thread)
#add_subdirectory(sanitizer/undefined)
#add_subdirectory(sanitizer/gprof)
#add_subdirectory(sanitizer/profile)