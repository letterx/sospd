#ifndef _SVM_STRUCT_OPTIONS_HPP_
#define _SVM_STRUCT_OPTIONS_HPP_

#include "svm_struct_api.h"
#include <string>

class SVM_App_Base;

SVM_App_Base* ParseStructLearnParameters(STRUCT_LEARN_PARM* sparm);
void PrintStructLearnHelp();

SVM_App_Base* ParseStructClassifyParameters(STRUCT_LEARN_PARM* sparm);
void PrintStructClassifyHelp();

SVM_App_Base* ParseFeatureTrainParameters(int argc, char** argv, std::string& train_file, std::string& eval_file, std::string& output_dir);

#endif
