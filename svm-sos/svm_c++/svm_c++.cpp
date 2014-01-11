#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
extern "C" {
#include "svm_struct/svm_struct_common.h"
#include "svm_struct_api.h"
}
#include "svm_c++.hpp"

std::unique_ptr<SVM_Cpp_Base> g_application{};

void SVM_Cpp_Base::trainFeatures(const std::string& train_file, 
        const std::string& eval_file, 
        const std::string& output_dir) {
    PatternVec train_patterns;
    LabelVec train_labels;
    PatternVec eval_patterns;
    LabelVec eval_labels;
    readExamples(train_file, train_patterns, train_labels);
    readExamples(eval_file, eval_patterns, eval_labels);
    //initFeatures(m_derived->Params());

    for (const auto& fgp : features()) {
        fgp->Train(train_patterns, train_labels);
    }
    for (const auto& fgp : features()) {
        fgp->Evaluate(eval_patterns);
        fgp->SaveEvaluation(output_dir);
    }
}

long SVM_Cpp_Base::numFeatures() const {
    long n = 0;
    for (const auto& fgp : features())
        n += fgp->NumFeatures();
    return n;
}
