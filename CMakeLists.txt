cmake_minimum_required(VERSION 2.8)
project(hseg)

option(FULL_OPTIMIZE "Enable full optimization" ON)

set(optimization_flags "-O3 -DEIGEN_NO_DEBUG")
if (FULL_OPTIMIZE)
    set(optimization_flags "${optimization_flags} -march=native -mfpmath=sse")
endif ()

### Full optimization ###
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${optimization_flags}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO  "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${optimization_flags} -fno-omit-frame-pointer")
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

# Configure cmake to also look for non-standard modules in the local directory
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

### Need threading capabilities ###
find_package(Threads REQUIRED)

### BOOST ###
set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.58 COMPONENTS filesystem system REQUIRED)

### OpenCV ###
find_package(OpenCV REQUIRED)

### Eigen ###
find_package(Eigen3 REQUIRED)

### libPNG ###
find_package(PNG REQUIRED)

### MatIO ###
find_package(MATIO REQUIRED)

### gco ###
add_subdirectory(lib/gco-v3.0)

### trw_s ###
add_subdirectory(lib/trw_s)

### properties ###
add_subdirectory(lib/properties)

### Set general gcc flags ###
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

### Function to prepend a string to all elements of a list
FUNCTION(PREPEND var prefix)
    SET(listVar "")
    FOREACH (f ${ARGN})
        LIST(APPEND listVar "${prefix}/${f}")
    ENDFOREACH (f)
    SET(${var} "${listVar}" PARENT_SCOPE)
ENDFUNCTION(PREPEND)

### All the source files and headers
set(SOURCE_FILES include/Image/Image.h include/helper/coordinate_helper.h include/helper/image_helper.h src/helper/image_helper.cpp include/helper/opencv_helper.h src/helper/opencv_helper.cpp src/Energy/EnergyFunction.cpp include/Energy/EnergyFunction.h include/Image/Coordinates.h src/Energy/Weights.cpp include/Energy/Weights.h include/helper/hash_helper.h src/Timer.cpp include/Timer.h src/Accuracy/ConfusionMatrix.cpp include/Accuracy/ConfusionMatrix.h include/Inference/InferenceIterator.h include/Inference/InferenceResult.h include/Inference/InferenceResultDetails.h src/Threading/ThreadPool.cpp include/Threading/ThreadPool.h include/typedefs.h src/Image/FeatureImage.cpp include/Image/FeatureImage.h include/Image/Feature.h src/Energy/LossAugmentedEnergyFunction.cpp include/Energy/LossAugmentedEnergyFunction.h include/Inference/Cluster.h include/helper/clustering_helper.h src/helper/clustering_helper.cpp)
set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(INCLUDE_SYS_DIRS ${gco_INCLUDE_DIRS} ${trw_s_INCLUDE_DIRS} ${properties_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR} ${PNG_INCLUDE_DIRS} ${MATIO_INCLUDE_DIRS})
set(LIBS gco trw_s properties ${OpenCV_LIBS} ${Boost_LIBRARIES} ${PNG_LIBRARIES} ${MATIO_LIBRARIES})
PREPEND(SOURCE_FILES_FULL_PATH ${CMAKE_CURRENT_SOURCE_DIR} ${SOURCE_FILES})

add_definitions(${PNG_DEFINITIONS})

### Compile main library ###
add_library(hseg STATIC ${SOURCE_FILES})
target_include_directories(hseg PUBLIC ${INCLUDE_DIRS})
target_include_directories(hseg SYSTEM PUBLIC  ${INCLUDE_SYS_DIRS})
target_link_libraries(hseg ${LIBS})
set_target_properties(hseg PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Wpedantic")
if(THREADS_HAVE_PTHREAD_ARG)
    target_compile_options(PUBLIC hseg "-pthread")
endif()
if(CMAKE_THREAD_LIBS_INIT)
    target_link_libraries(hseg "${CMAKE_THREAD_LIBS_INIT}")
endif()

### Compile inference executables ###
add_executable(hseg_infer exec/inference.cpp)
target_include_directories(hseg_infer PUBLIC ${INCLUDE_DIRS})
target_include_directories(hseg_infer SYSTEM PUBLIC  ${INCLUDE_SYS_DIRS})
target_link_libraries(hseg_infer ${LIBS} hseg)
set_target_properties(hseg_infer PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Wpedantic")
add_executable(hseg_infer_batch exec/inference_batch.cpp)
target_include_directories(hseg_infer_batch PUBLIC ${INCLUDE_DIRS})
target_include_directories(hseg_infer_batch SYSTEM PUBLIC  ${INCLUDE_SYS_DIRS})
target_link_libraries(hseg_infer_batch ${LIBS} hseg)
set_target_properties(hseg_infer_batch PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Wpedantic")

### Compile accuracy executable ###
add_executable(hseg_accy exec/accuracy.cpp)
target_include_directories(hseg_accy PUBLIC ${INCLUDE_DIRS})
target_include_directories(hseg_accy SYSTEM PUBLIC  ${INCLUDE_SYS_DIRS})
target_link_libraries(hseg_accy ${LIBS} hseg)
set_target_properties(hseg_accy PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Wpedantic")

### Compile training executable ###
add_executable(hseg_train exec/train.cpp)
target_include_directories(hseg_train PUBLIC ${INCLUDE_DIRS})
target_include_directories(hseg_train SYSTEM PUBLIC  ${INCLUDE_SYS_DIRS})
target_link_libraries(hseg_train ${LIBS} hseg)
set_target_properties(hseg_train PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Wpedantic")

### Compile utility executable ###
add_executable(hseg_util exec/util.cpp)
target_include_directories(hseg_util PUBLIC ${INCLUDE_DIRS})
target_include_directories(hseg_util SYSTEM PUBLIC  ${INCLUDE_SYS_DIRS})
target_link_libraries(hseg_util ${LIBS} hseg)
set_target_properties(hseg_util PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Wpedantic")

### Compile distributed training executables ###
add_executable(hseg_train_dist_pred exec/train_dist_pred.cpp)
target_include_directories(hseg_train_dist_pred PUBLIC ${INCLUDE_DIRS})
target_include_directories(hseg_train_dist_pred SYSTEM PUBLIC  ${INCLUDE_SYS_DIRS})
target_link_libraries(hseg_train_dist_pred ${LIBS} hseg)
set_target_properties(hseg_train_dist_pred PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Wpedantic")
add_executable(hseg_train_dist_merge exec/train_dist_merge.cpp)
target_include_directories(hseg_train_dist_merge PUBLIC ${INCLUDE_DIRS})
target_include_directories(hseg_train_dist_merge SYSTEM PUBLIC  ${INCLUDE_SYS_DIRS})
target_link_libraries(hseg_train_dist_merge ${LIBS} hseg)
set_target_properties(hseg_train_dist_merge PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Wpedantic")

### Compile messy executable (i.e. the one where I do all my testing) ###
add_executable(hseg_mess exec/messy.cpp)
target_include_directories(hseg_mess PUBLIC ${INCLUDE_DIRS})
target_include_directories(hseg_mess SYSTEM PUBLIC  ${INCLUDE_SYS_DIRS})
target_link_libraries(hseg_mess ${LIBS} hseg)

### Unit tests ###
enable_testing()
find_package(GTest)
if (GTest_FOUND)
    set(TEST_SOURCE_FILES test/Inference/InferenceIterator_Test.cpp)
    add_executable(hseg_test test/gtest.cpp ${TEST_SOURCE_FILES})
    target_include_directories(hseg_test PUBLIC ${GTEST_INCLUDE_DIRS} test ${INCLUDE_DIRS})
    target_link_libraries(hseg_test ${GTEST_BOTH_LIBRARIES} ${LIBS} hseg)
    add_test(AllTests hseg_test)
else ()
    message(WARNING "GTest not found, not building unit tests.")
endif ()