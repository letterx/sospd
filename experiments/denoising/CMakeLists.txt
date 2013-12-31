###
### Denoise executable
###

set(sources
    "denoise.cpp"
    "foe-cliques.cpp"
)

add_executable(denoise ${sources})

find_package(OpenCV REQUIRED)

set(libs ${libs} boost_program_options ${OpenCV_LIBS})

target_link_libraries(denoise "sos-opt" ${libs})

###
### Target: add-noise
###

add_executable(add-noise add-noise.cpp)

target_link_libraries(add-noise sos-opt ${OpenCV_LIBS} ${libs})
