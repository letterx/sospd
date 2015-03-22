###
### Stereo executable
###

add_executable(stereo "stereo.cpp")

find_package(OpenCV REQUIRED)

set(libs ${libs} boost_program_options ${OpenCV_LIBS})

target_link_libraries(stereo "sos-opt" ${libs})

if(WITH_GUROBI)
    target_link_libraries(stereo "${GUROBI_LIBRARY}")
endif()


###
### disparity-diff executable
###

add_executable(disparity-diff "disparity-diff.cpp")

target_link_libraries(disparity-diff ${OpenCV_LIBS})

###
### stereo-to-opengm executable
###

if(WITH_OPENGM)
    add_executable(stereo-to-opengm "stereo-to-opengm.cpp")
    target_link_libraries(stereo-to-opengm ${libs} ${HDF5_LIBRARIES})

    add_executable(opengm-to-stereo "opengm-to-stereo.cpp")
    target_link_libraries(opengm-to-stereo "sos-opt" ${libs})
    if (WITH_GUROBI)
        target_link_libraries(opengm-to-stereo "${GUROBI_LIBRARY}")
    endif()
endif(WITH_OPENGM)
