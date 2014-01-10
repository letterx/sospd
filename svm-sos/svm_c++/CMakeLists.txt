add_library(svm_cpp_common STATIC
    "svm_struct/svm_struct_common.c"
    "svm_light/svm_hideo.c"
    "svm_light/svm_learn.c"
    "svm_light/svm_common.c"
    "svm_struct_learn_custom.cpp"
    "svm_struct_api.cpp"
    )

add_library(svm_cpp_learn 
    "svm_struct/svm_struct_learn.c"
    "svm_struct/svm_struct_main.c"
    )

add_library(svm_cpp_classify STATIC
    "svm_struct/svm_struct_classify.c"
    )
    
