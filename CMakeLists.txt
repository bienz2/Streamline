cmake_minimum_required(VERSION 2.8)
project(node_aware_mpi)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

enable_language(CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)

option(CRAY "Running on Cray Machine" OFF)
option(BGQ "Running on BGQ" OFF)
if (CRAY)
    add_definitions (-DUSING_CRAY)
elseif (BGQ)
    add_definitions(-DUSING_BGQ)
endif()

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

set(GOOGLETEST_ROOT googletest CACHE STRING "Google Test source root")
include_directories(
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}/include
    )
set(GOOGLETEST_SOURCES
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}/src/gtest-all.cc
    ${PROJECT_SOURCE_DIR}/${GOOGLETEST_ROOT}/src/gtest_main.cc
    )
foreach(_source ${GOOGLETEST_SOURCES})
    set_source_files_properties(${_source} PROPERTIES GENERATED 1)
endforeach()
add_library(googletest STATIC ${GOOGLETEST_SOURCES})

include_directories(".")
enable_testing()
add_subdirectory(tests)

