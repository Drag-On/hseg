cmake_minimum_required(VERSION 2.8)
project(trw_s)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")

set(trw_s_INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR} CACHE PATH "Include directory for the trw_s library")
set(TRW_S_SOURCE instances.h minimize.cpp MRFEnergy.h MRFEnergy.cpp ordering.cpp treeProbabilities.cpp typeBinary.h typeBinaryFast.h typeGeneral.h typePotts.h typeTruncatedLinear.h typeTruncatedLinear2D.h typeTruncatedQuadratic.h typeTruncatedQuadratic2D.h)
add_library(trw_s ${trw_s_INCLUDE_DIRS} ${TRW_S_SOURCE})

set_source_files_properties(
        ${TRW_S_SOURCE}
        PROPERTIES
        COMPILE_FLAGS "-Wno-cpp"
)
