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
