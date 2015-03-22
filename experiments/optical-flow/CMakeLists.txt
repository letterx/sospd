###
### Denoise executable
###

set(sources
    "flow.cpp"
)

add_executable(flow ${sources})

find_package(OpenCV REQUIRED)

set(libs ${libs} boost_program_options ${OpenCV_LIBS})

target_link_libraries(flow "sos-opt" ${libs})

if(WITH_GUROBI)
    target_link_libraries(flow "${GUROBI_LIBRARY}")
endif()
