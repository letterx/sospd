#include "svm_struct_options.hpp"
#include <boost/program_options.hpp>

bool global_show_images = false;

void ParseStructLearnParameters(STRUCT_LEARN_PARM* sparm) {
    int i;

    sparm->grabcut_classify = 0;
    sparm->crf = 0;
    sparm->pairwise_feature = 0;
    sparm->contrast_pairwise_feature = 0;
    sparm->submodular_feature = 1;

    for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2]) 
      { 
      case 'c': i++; sparm->crf=atol(sparm->custom_argv[i]); break;
      case 'p': i++; sparm->pairwise_feature=atol(sparm->custom_argv[i]); break;
      case 'g': i++; sparm->contrast_pairwise_feature=atol(sparm->custom_argv[i]); break;
      case 's': i++; sparm->submodular_feature=atol(sparm->custom_argv[i]); break;
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
           exit(0);
      }
    }
}

void PrintStructLearnHelp() {

}

void ParseStructClassifyParameters(STRUCT_LEARN_PARM* sparm) {
    int i;

    sparm->grabcut_classify = 0;

    for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2]) 
      { 
      /* case 'x': i++; strcpy(xvalue,sparm->custom_argv[i]); break; */
      case 'g': i++; sparm->grabcut_classify=atol(sparm->custom_argv[i]); break;
      case 's': i++; global_show_images=atol(sparm->custom_argv[i]); break;
      case 'c': i++; sparm->crf=atol(sparm->custom_argv[i]); break;
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
           exit(0);
      }
    }
}

void PrintStructClassifyHelp() {

}
