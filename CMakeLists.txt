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
#set(Boost_USE_STATIC_LIBS ON)
find_package(Boost 1.58 COMPONENTS filesystem system REQUIRED)

### OpenCV ###
find_package(OpenCV REQUIRED)

### Eigen ###
find_package(Eigen3 REQUIRED)

### libPNG ###
find_package(PNG REQUIRED)

### MatIO ###
find_package(MATIO REQUIRED)

### trw_s ###
add_subdirectory(lib/trw_s)

### properties ###
add_subdirectory(lib/properties)

### Set general gcc flags ###
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

### caffe ###
add_subdirectory(lib/caffe)

add_definitions(${PNG_DEFINITIONS})

### Compile main library ###
set(ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
include(cmake/compile_hseg.cmake)

### Compile inference executables ###
add_executable(hseg_infer exec/inference.cpp)
target_link_libraries(hseg_infer ${LIBS} hseg)
set_target_properties(hseg_infer PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Wpedantic")
add_executable(hseg_infer_batch exec/inference_batch.cpp)
target_link_libraries(hseg_infer_batch hseg)
set_target_properties(hseg_infer_batch PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Wpedantic")

### Compile accuracy executable ###
add_executable(hseg_accy exec/accuracy.cpp)
target_link_libraries(hseg_accy hseg)
set_target_properties(hseg_accy PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Wpedantic")

### Compile training executable ###
add_executable(hseg_train exec/train.cpp)
target_link_libraries(hseg_train hseg)
set_target_properties(hseg_train PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Wpedantic")

### Compile utility executable ###
add_executable(hseg_util exec/util.cpp)
target_include_directories(hseg_util SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/lib/caffe/include ${CMAKE_BINARY_DIR}/lib/caffe/include)
target_link_libraries(hseg_util hseg caffe)
set_target_properties(hseg_util PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Wpedantic")

### Compile distributed training executables ###
add_executable(hseg_train_dist_pred exec/train_dist_pred.cpp)
target_link_libraries(hseg_train_dist_pred ${LIBS} hseg)
set_target_properties(hseg_train_dist_pred PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Wpedantic")
add_executable(hseg_train_dist_merge exec/train_dist_merge.cpp)
target_link_libraries(hseg_train_dist_merge hseg)
set_target_properties(hseg_train_dist_merge PROPERTIES COMPILE_FLAGS "-Wall -Wextra -Wpedantic")

### Compile messy executable (i.e. the one where I do all my testing) ###
add_executable(hseg_mess exec/messy.cpp)
target_link_libraries(hseg_mess hseg)

### Unit tests ###
enable_testing()
find_package(GTest)
if (GTest_FOUND)
    set(TEST_SOURCE_FILES test/Inference/InferenceIterator_Test.cpp)
    add_executable(hseg_test test/gtest.cpp ${TEST_SOURCE_FILES})
    target_include_directories(hseg_test PUBLIC ${GTEST_INCLUDE_DIRS} test)
    target_link_libraries(hseg_test ${GTEST_BOTH_LIBRARIES} ${LIBS} hseg)
    add_test(AllTests hseg_test)
else ()
    message(WARNING "GTest not found, not building unit tests.")
endif ()