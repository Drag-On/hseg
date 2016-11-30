cmake_minimum_required(VERSION 2.8)
project(hseg)

### Full optimization ###
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mfpmath=sse")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -march=native -mfpmath=sse -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer")
# Note: -fno-omit-frame-pointer is needed by Zoom profiler. It should not have any impact on performance, however I only
#       include it in Debug and RelWithDebInfo mode (the latter of which should be used for profiling).

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

### Need threading capabilities ###
find_package(Threads REQUIRED)

### BOOST ###
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.58 COMPONENTS filesystem system REQUIRED)

### OpenCV ###
find_package(OpenCV REQUIRED)

### Eigen ###
find_package(Eigen3 REQUIRED)

### gco ###
add_subdirectory(lib/gco-v3.0)

### trw_s ###
add_subdirectory(lib/trw_s)

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

set(SOURCE_FILES src/Energy/UnaryFile.cpp include/Energy/UnaryFile.h include/Inference/k-prototypes/Clusterer.h include/Image/Image.h include/Inference/k-prototypes/Cluster.h include/helper/coordinate_helper.h include/Inference/k-prototypes/Feature.h src/Inference/k-prototypes/Feature.cpp src/Inference/k-prototypes/Cluster.cpp include/helper/image_helper.h src/helper/image_helper.cpp include/helper/opencv_helper.h src/helper/opencv_helper.cpp include/Inference/GraphOptimizer/GraphOptimizer.h src/Energy/EnergyFunction.cpp include/Energy/EnergyFunction.h include/Image/Coordinates.h src/Energy/WeightsVec.cpp include/Energy/WeightsVec.h include/helper/hash_helper.h src/Timer.cpp include/Timer.h src/Accuracy/ConfusionMatrix.cpp include/Accuracy/ConfusionMatrix.h include/Inference/InferenceIterator.h include/Inference/InferenceResult.h include/Inference/InferenceResultDetails.h src/Energy/LossAugmentedEnergyFunction.cpp include/Energy/LossAugmentedEnergyFunction.h src/Threading/ThreadPool.cpp include/Threading/ThreadPool.h include/Energy/feature_weights.h src/Energy/feature_weights.cpp include/Inference/TRW_S_Optimizer/TRW_S_Optimizer.h)
set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include ${gco_INCLUDE_DIRS} ${trw_s_INCLUDE_DIRS} ${properties_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
set(LIBS gco trw_s properties ${OpenCV_LIBS} ${Boost_LIBRARIES})
PREPEND(SOURCE_FILES_FULL_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCE_FILES})

### Compile main library ###
add_library(hseg ${SOURCE_FILES})
target_include_directories(hseg PUBLIC ${INCLUDE_DIRS})
target_link_libraries(hseg ${LIBS})
if(THREADS_HAVE_PTHREAD_ARG)
    target_compile_options(PUBLIC hseg "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(hseg "${CMAKE_THREAD_LIBS_INIT}")
endif()

### Compile clustering executable ###
add_executable(hseg_cluster exec/clustering.cpp)
target_include_directories(hseg_cluster PUBLIC ${INCLUDE_DIRS})
target_link_libraries(hseg_cluster ${LIBS} hseg)

### Compile inference executables ###
add_executable(hseg_infer exec/inference.cpp)
target_include_directories(hseg_infer PUBLIC ${INCLUDE_DIRS})
target_link_libraries(hseg_infer ${LIBS} hseg)
add_executable(hseg_infer_batch exec/inference_batch.cpp)
target_include_directories(hseg_infer_batch PUBLIC ${INCLUDE_DIRS})
target_link_libraries(hseg_infer_batch ${LIBS} hseg)

### Compile accuracy executable ###
add_executable(hseg_accy exec/accuracy.cpp)
target_include_directories(hseg_accy PUBLIC ${INCLUDE_DIRS})
target_link_libraries(hseg_accy ${LIBS} hseg)

### Compile training executable ###
add_executable(hseg_train exec/train.cpp)
target_include_directories(hseg_train PUBLIC ${INCLUDE_DIRS})
target_link_libraries(hseg_train ${LIBS} hseg)

### Compile utility executable ###
add_executable(hseg_util exec/util.cpp)
target_include_directories(hseg_util PUBLIC ${INCLUDE_DIRS})
target_link_libraries(hseg_util ${LIBS} hseg)

### Compile distributed training executables ###
add_executable(hseg_train_dist_pred exec/train_dist_pred.cpp)
target_include_directories(hseg_train_dist_pred PUBLIC ${INCLUDE_DIRS})
target_link_libraries(hseg_train_dist_pred ${LIBS} hseg)
add_executable(hseg_train_dist_merge exec/train_dist_merge.cpp)
target_include_directories(hseg_train_dist_merge PUBLIC ${INCLUDE_DIRS})
target_link_libraries(hseg_train_dist_merge ${LIBS} hseg)

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
