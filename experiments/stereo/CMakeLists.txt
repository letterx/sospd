###
### Stereo executable
###

add_executable(stereo "stereo.cpp")

find_package(OpenCV REQUIRED)

set(libs ${libs} boost_program_options ${OpenCV_LIBS})

target_link_libraries(stereo "sos-opt" ${libs})


###
### disparity-diff executable
###

add_executable(disparity-diff "disparity-diff.cpp")

target_link_libraries(disparity-diff ${OpenCV_LIBS})
