cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-generate=.profiling")
add_executable(hseg_profile EXCLUDE_FROM_ALL ${SOURCE_FILES_FULL_PATH})
target_include_directories(hseg_profile PUBLIC ${INCLUDE_DIRS})
target_link_libraries(hseg_profile ${LIBS})