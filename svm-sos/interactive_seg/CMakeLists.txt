set(interactive_srcs
    gmm.cpp
    interactive_seg.cpp
    )

add_library(interactive_seg STATIC ${interactive_srcs})
add_executable(interactive_seg_learn dummy.cpp)
add_executable(interactive_seg_feature dummy.cpp)
add_executable(interactive_seg_classify dummy.cpp)

set(interactive_libs
    ${sos_svm_libs}
    svm_cpp_common
    opencv_core
    opencv_highgui
    opencv_imgproc
    )


target_link_libraries(interactive_seg_learn
    interactive_seg
    svm_cpp_learn
    ${interactive_libs}
    )

target_link_libraries(interactive_seg_feature
    interactive_seg
    svm_cpp_feature
    ${interactive_libs}
    )

target_link_libraries(interactive_seg_classify
    interactive_seg
    ${interactive_libs}
    svm_cpp_classify
    )
