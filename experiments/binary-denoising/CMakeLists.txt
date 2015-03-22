###
### Denoise executable
###

set(sources
    "binary-denoise.cpp"
)

add_executable(binary-denoise ${sources})

find_package(OpenCV REQUIRED)

set(libs ${libs} boost_program_options ${OpenCV_LIBS})

target_link_libraries(binary-denoise "sos-opt" ${libs})

if(WITH_GUROBI)
    target_link_libraries(binary-denoise "${GUROBI_LIBRARY}")
endif()

###
### Train executable
###
add_executable(binary-train "binary-train.cpp")

target_link_libraries(binary-train ${libs})

add_executable(binary-add-noise "binary-add-noise.cpp")
target_link_libraries(binary-add-noise ${libs})

if(WITH_OPENGM)
    add_executable(binary-to-opengm "binary-to-opengm.cpp")
    target_link_libraries(binary-to-opengm ${libs} ${HDF5_LIBRARIES})
endif(WITH_OPENGM)
