/***********************************************************************/
/*                                                                     */
/*   svm_struct_api.c                                                  */
/*                                                                     */
/*   Definition of API for attaching implementing SVM learning of      */
/*   structures (e.g. parsing, multi-label classification, HMM)        */ 
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 03.07.04                                                    */
/*                                                                     */
/*   Copyright (c) 2004  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/***********************************************************************/

#include <stdio.h>
#include <string.h>
extern "C" {
#include "svm_struct/svm_struct_common.h"
#include "svm_struct_api.h"
}
#include "svm_c++.hpp"
#include "svm_struct_options.hpp"

void        svm_struct_learn_api_init(int argc, char* argv[])
{
  /* Called in learning part before anything else is done to allow
     any initializations that might be necessary. */
}

void        svm_struct_learn_api_exit()
{
  /* Called in learning part at the very end to allow any clean-up
     that might be necessary. */
    g_application->svm_struct_learn_api_exit();
    delete g_application;
}

void        svm_struct_classify_api_init(int argc, char* argv[])
{
  /* Called in prediction part before anything else is done to allow
     any initializations that might be necessary. */
}

void        svm_struct_classify_api_exit()
{
  /* Called in prediction part at the very end to allow any clean-up
     that might be necessary. */
    g_application->svm_struct_classify_api_exit();
    delete g_application;
}

SAMPLE      read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm)
{
    return g_application->read_struct_examples(file, sparm);
}

void        init_struct_model(SAMPLE sample, STRUCTMODEL *sm, 
			      STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, 
			      KERNEL_PARM *kparm)
{
    g_application->init_struct_model(sample, sm, sparm, lparm, kparm);
}

CONSTSET    init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm)
{
    return g_application->init_struct_constraints(sample, sm, sparm);
}

LABEL       classify_struct_example(PATTERN x, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm)
{
    return g_application->classify_struct_example(x, sm, sparm);
}

LABEL       find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm)
{
    return g_application->find_most_violated_constraint_slackrescaling(x, y, sm, sparm);
}

LABEL       find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm)
{
    return g_application->find_most_violated_constraint_marginrescaling(x, y, sm, sparm);
}

int         empty_label(LABEL y)
{
    return g_application->empty_label(y);
}

SVECTOR     *psi(PATTERN x, LABEL y, STRUCTMODEL *sm,
		 STRUCT_LEARN_PARM *sparm)
{
    return g_application->psi(x, y, sm, sparm);
}

double      loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm)
{
    return g_application->loss(y, ybar, sparm);
}

int         finalize_iteration(double ceps, int cached_constraint,
			       SAMPLE sample, STRUCTMODEL *sm,
			       CONSTSET cset, double *alpha, 
			       STRUCT_LEARN_PARM *sparm)
{
    return g_application->finalize_iteration(ceps, cached_constraint, sample, sm, cset, alpha, sparm);
}

void        final_train_stats(double maxdiff, double epsilon, 
                   double modellength, double slacksum) {
    g_application->final_train_stats(maxdiff, epsilon, modellength, slacksum);
}

void        print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm,
					CONSTSET cset, double *alpha, 
					STRUCT_LEARN_PARM *sparm)
{
    g_application->print_struct_learning_stats(sample, sm, cset, alpha, sparm);
}

void        print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm,
				       STRUCT_LEARN_PARM *sparm, 
				       STRUCT_TEST_STATS *teststats)
{
    g_application->print_struct_testing_stats(sample, sm, sparm, teststats);
}

void        eval_prediction(long exnum, EXAMPLE ex, LABEL ypred, 
			    STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, 
			    STRUCT_TEST_STATS *teststats)
{
    g_application->eval_prediction(exnum, ex, ypred, sm, sparm, teststats);
}

void        write_struct_model(char *file, STRUCTMODEL *sm, 
			       STRUCT_LEARN_PARM *sparm)
{
    g_application->write_struct_model(file, sm, sparm);
}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm)
{
    return g_application->read_struct_model(file, sparm);
}

void        write_label(FILE* fp, LABEL y)
{
    g_application->write_label(fp, y);
} 

void        free_pattern(PATTERN x) {
    delete x.data;
}

void        free_label(LABEL y) {
    delete y.data;
}

void        free_struct_model(STRUCTMODEL sm) 
{
  /* Frees the memory of model. */
  /* if(sm.w) free(sm.w); */ /* this is free'd in free_model */
  if(sm.svm_model) free_model(sm.svm_model,1);
  /* add free calls for user defined data here */
}

void        free_struct_sample(SAMPLE s)
{
  /* Frees the memory of sample s. */
  int i;
  for(i=0;i<s.n;i++) { 
    free_pattern(s.examples[i].x);
    free_label(s.examples[i].y);
  }
  free(s.examples);
}

void        print_struct_help()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_learn. */
    PrintStructLearnHelp();
}

void         parse_struct_parameters(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- */
    g_application = ParseStructLearnParameters(sparm);
}

void        print_struct_help_classify()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_classify. */
    PrintStructClassifyHelp();
}

void         parse_struct_parameters_classify(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- for the
     classification module */
    g_application = ParseStructClassifyParameters(sparm);
}

