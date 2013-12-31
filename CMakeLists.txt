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
include_directories(${SOS_OPT_SOURCE_DIR}/higher-order-energy/include)
include_directories(${SOS_OPT_SOURCE_DIR}/higher-order-energy/qpbo)
include_directories(${SOS_OPT_SOURCE_DIR}/hocr)

set(lib-sources
    "src/submodular-ibfs.cpp"
    "src/submodular-flow.cpp"
    "src/gen-random.cpp"
    "src/alpha-expansion.cpp"
    "src/dgfm.cpp"
    "higher-order-energy/qpbo/QPBO.cpp"
)

find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

set(libs
    "boost_serialization")


###
### Build configuration
### 
set(CMAKE_BUILD_TYPE Debug)

# GCC settings
if(CMAKE_COMPILER_IS_GNUCXX)
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
	#
endif()

###
### Target: libsos-opt
###

add_library(sos-opt STATIC ${lib-sources})


###
### Subdirectories
###

add_subdirectory(experiments)
add_subdirectory(test)

###
### Target: higher-order-experiment
###

add_executable(higher-order-experiment higher-order-experiment.cpp)

target_link_libraries(higher-order-experiment sos-opt ${libs})


###
### Target: add-noise
###

add_executable(add-noise add-noise.cpp)

find_package(OpenCV REQUIRED)

target_link_libraries(add-noise sos-opt ${OpenCV_LIBS} ${libs})
