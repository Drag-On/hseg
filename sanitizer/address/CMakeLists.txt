cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
add_executable(hseg_addr EXCLUDE_FROM_ALL ${SOURCE_FILES_FULL_PATH})
target_include_directories(hseg_addr PUBLIC ${INCLUDE_DIRS})
target_link_libraries(hseg_addr ${LIBS})