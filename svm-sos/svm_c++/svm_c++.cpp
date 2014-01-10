#include <stdio.h>
#include <string.h>
#include <iostream>
#include <fstream>
extern "C" {
#include "svm_struct/svm_struct_common.h"
#include "svm_struct_api.h"
}
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include "energy-common.hpp"
#include "svm_c++.hpp"
#include "svm_struct_options.hpp"
#include "stats.hpp"

std::unique_ptr<SVM_Cpp_Base> g_application{};

void SVM_Cpp_Base::train_features(const std::string& train_file, 
        const std::string& eval_file, 
        const std::string& output_dir) {
    PatternVec train_patterns;
    LabelVec train_labels;
    PatternVec eval_patterns;
    LabelVec eval_labels;
    readExamples(train_file, train_patterns, train_labels);
    readExamples(eval_file, eval_patterns, eval_labels);
    initFeatures(m_derived->Params());

    for (auto fgp : features()) {
        fgp->Train(train_patterns, train_labels);
    }
    for (auto fgp : features()) {
        fgp->Evaluate(eval_patterns);
        fgp->SaveEvaluation(output_dir);
    }
}
template <class Derived>
STRUCTMODEL SVM_App<Derived>::read_struct_model(char *file, STRUCT_LEARN_PARM *sparm)
{
}

