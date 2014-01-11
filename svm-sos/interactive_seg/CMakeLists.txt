add_executable(interactive_seg_learn
    gmm.cpp
    interactive_seg.cpp
    )

target_link_libraries(interactive_seg_learn
    svm_cpp_learn
    svm_cpp_common
    ${sos_svm_libs}
    )
