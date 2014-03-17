set(svm_struct_source
    "svm_struct/svm_struct_common.c"
    "svm_light/svm_hideo.c"
    "svm_light/svm_learn.c"
    "svm_light/svm_common.c"
    )

set_source_files_properties(${svm_struct_source} PROPERTIES
    COMPILE_FLAGS "-w -ffast-math")

add_library(svm_cpp_common STATIC
    ${svm_struct_source}
    "svm_struct_learn_custom.cpp"
    "svm_struct_api.cpp"
    "svm_c++.cpp"
    "stats.cpp"
    )

add_library(svm_cpp_learn 
    "svm_struct/svm_struct_learn.c"
    "svm_struct/svm_struct_main.c"
    )

add_library(svm_cpp_classify 
    "svm_struct/svm_struct_classify.c"
    )
    
add_library(svm_cpp_feature STATIC
    feature_train.cpp
    )
