###
### CMake settings
###
cmake_minimum_required(VERSION 2.6)
cmake_policy(SET CMP0012 NEW)


###
### Project settings
###
project(SOS_OPT)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules)
set(SOS_OPT_VERSION_MAJOR "1")
set(SOS_OPT_VERSION_MINOR "1")
set(SOS_OPT_VERSION_PATCH "0")
set(SOS_OPT_VERSION "${SOS_OPT_VERSION_MAJOR}.${SOS_OPT_VERSION_MINOR}.${SOS_OPT_VERSION_PATCH}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

OPTION(WITH_GRD "Include GRD" OFF)
OPTION(WITH_GUROBI "Include GUROBI" OFF)
OPTION(WITH_OPENGM "Include OpenGM" OFF)

###
### Sources, headers, directories and libs
###
include_directories(${SOS_OPT_SOURCE_DIR}/include)
include_directories(${SOS_OPT_SOURCE_DIR}/src)
include_directories(thirdparty)
include_directories(thirdparty/QPBO)
include_directories(thirdparty/HOCR)

set(lib-sources
    "src/bidirectional-ibfs.cpp"
    "src/gen-random.cpp"
    "src/parametric-ibfs.cpp"
    "src/sospd.cpp"
    "src/source-ibfs.cpp"
    "src/submodular-functions.cpp"
    "src/submodular-ibfs.cpp"
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

if (WITH_GRD)
    set(petter-libs
        pseudoboolean
        petter
        maxflow3
        QPBO
        Clp
        CoinUtils
        )

    set(libs
        ${petter-libs}
        )
else()
    message(STATUS "build without GRD")
endif()

if(WITH_GUROBI)
   message(STATUS "build with GUROBI interface")
   find_package(GUROBI REQUIRED)
   add_definitions(-DWITH_GUROBI)
   include_directories(${GUROBI_INCLUDE_DIRS})
else()
   message(STATUS "build without GUROBI interface")
endif()

if (WITH_OPENGM)
    message(STATUS "build with opengm")
    SET(OPENGM_INCLUDE_DIR "" CACHE STRING "Include directory for OpenGM")
    SET(OPENGM_LIB CACHE STRING "Library for OpenGM")
    include_directories(${OPENGM_INCLUDE_DIR})
    add_definitions(-DWITH_OPENGM)

    message(STATUS "build with HDF5 support")
    if(WIN32)
      # FindHDF5 is broken on Windows
      # it wrongly discovers the dlls instead of the lib files
      # see: www.cmake.org/Bug/bug_relationship_graph.php?bug_id=14111
      # therefore we set the variable manually
      find_library( HDF5_IMPORT_LIB NAMES hdf5 hdf5dll )
      find_library( HDF5_HL_IMPORT_LIB NAMES hdf5_hl hdf5_hldll )
      find_path(HDF5_INCLUDE_DIR hdf5.h)
      set( HDF5_LIBRARIES ${HDF5_IMPORT_LIB} ${HDF5_HL_IMPORT_LIB} )
      message(STATUS "HDF5: autodiscovery is broken on WIN32; using global search paths")
      message(STATUS "HDF5: found ${HDF5_LIBRARIES}")
    else()
      find_package(HDF5 REQUIRED)
    endif()
    include_directories(${HDF5_INCLUDE_DIR})
    add_definitions(${HDF5_CPPFLAGS})
    add_definitions(-DWITH_HDF5)
else(WITH_OPENGM)
    message(STATUS "build without OpenGM support")
endif(WITH_OPENGM)


###
### Build configuration
### 
#set(CMAKE_BUILD_TYPE Debug)

# GCC settings
#
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O3 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os")
#
set(GCC_WARNING_FLAGS "-Wall -Wextra -pedantic -Wno-missing-braces -Woverloaded-virtual -Wno-unused-parameter")
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

