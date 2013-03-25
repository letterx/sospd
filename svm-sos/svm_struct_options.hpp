#ifndef _SVM_STRUCT_OPTIONS_HPP_
#define _SVM_STRUCT_OPTIONS_HPP_

#include "svm_struct_api.h"

void ParseStructLearnParameters(STRUCT_LEARN_PARM* sparm);
void PrintStructLearnHelp();

void ParseStructClassifyParameters(STRUCT_LEARN_PARM* sparm);
void PrintStructClassifyHelp();

#endif
