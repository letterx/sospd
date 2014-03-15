###
### CMake settings
###
cmake_minimum_required(VERSION 2.6)
cmake_policy(SET CMP0012 NEW)


###
### Project settings
###
project(SOS_OPT)

set(SOS_OPT_VERSION_MAJOR "1")
set(SOS_OPT_VERSION_MINOR "0")
set(SOS_OPT_VERSION_PATCH "0")
set(SOS_OPT_VERSION "${SOS_OPT_VERSION_MAJOR}.${SOS_OPT_VERSION_MINOR}.${SOS_OPT_VERSION_PATCH}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})


###
### Sources, headers, directories and libs
###
include_directories(${SOS_OPT_SOURCE_DIR}/include)
include_directories(${SOS_OPT_SOURCE_DIR}/src)
include_directories(thirdparty/QPBO)
include_directories(thirdparty/HOCR)

set(lib-sources
    "src/submodular-ibfs.cpp"
    "src/submodular-flow.cpp"
    "src/gen-random.cpp"
    "src/dgfm.cpp"
)

set(qpbo-sources
    "thirdparty/QPBO/QPBO.cpp"
    "thirdparty/QPBO/QPBO_extra.cpp"
    "thirdparty/QPBO/QPBO_maxflow.cpp"
    "thirdparty/QPBO/QPBO_postprocessing.cpp"
)

set(lib-sources ${lib-sources} ${qpbo-sources})

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(libs
    "boost_serialization")


###
### Build configuration
### 
set(CMAKE_BUILD_TYPE Debug)

# GCC settings
#
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os")
#
set(GCC_WARNING_FLAGS "-Wall -Wextra -pedantic -Wold-style-cast -Woverloaded-virtual -Wno-unused-parameter")
#set(GCC_WARNING_FLAGS "${GCC_WARNING_FLAGS} -Werror")
#
set(CMAKE_CXX_FLAGS "--std=c++11 ${GCC_WARNING_FLAGS} ${CMAKE_CXX_FLAGS}")
set_source_files_properties(${qpbo-sources} PROPERTIES COMPILE_FLAGS "-w")
#

###
### Target: libsos-opt
###

add_library(sos-opt STATIC ${lib-sources})


###
### Subdirectories
###

add_subdirectory(thirdparty)
add_subdirectory(experiments)
add_subdirectory(test)
add_subdirectory(svm-sos)

###
### Target: higher-order-experiment
###

add_executable(higher-order-experiment higher-order-experiment.cpp)

target_link_libraries(higher-order-experiment sos-opt ${libs})

