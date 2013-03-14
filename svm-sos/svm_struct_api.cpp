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
#include "image_manip.hpp"

PatternData* data(PATTERN& p) { return (PatternData*)p.data; }
PATTERN MakePattern(PatternData* d) {
    PATTERN p;
    p.data = d;
    return p;
}
LabelData* data(LABEL& l) { return (LabelData*)l.data; }
LABEL MakeLabel(LabelData* d) {
    LABEL l;
    l.data = d;
    return l;
}
ModelData* data(STRUCTMODEL& sm) { return (ModelData*)sm.data; }
ModelData* data(STRUCTMODEL* sm) { return (ModelData*)sm->data; }

void        svm_struct_learn_api_init(int argc, char* argv[])
{
  /* Called in learning part before anything else is done to allow
     any initializations that might be necessary. */
}

void        svm_struct_learn_api_exit()
{
  /* Called in learning part at the very end to allow any clean-up
     that might be necessary. */
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
}

SAMPLE      read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads struct examples and returns them in sample. The number of
     examples must be written into sample.n */
  SAMPLE   sample;  /* sample */
  EXAMPLE  *examples;
  size_t n = 0;       /* number of examples */

  std::ifstream main_file(file);

  std::string images_dir;
  std::string trimap_dir;
  std::string gt_dir;
  std::string line;

  do {
      std::getline(main_file, images_dir);
  } while (images_dir[0] == '#');
  do {
      std::getline(main_file, trimap_dir);
  } while (trimap_dir[0] == '#');
  do {
      std::getline(main_file, gt_dir);
  } while (gt_dir[0] == '#');

  std::vector<PatternData*> patterns;
  std::vector<LabelData*> labels;

  while (main_file.good()) {
      std::getline(main_file, line);
      if (!line.empty() && line[0] != '#') {
          n++;
          if (n % 10 == 0) {
              std::cout << ".";
              std::cout.flush();
          }
          cv::Mat image = cv::imread(images_dir + line, CV_LOAD_IMAGE_COLOR);
          cv::Mat trimap = cv::imread(trimap_dir + line, CV_LOAD_IMAGE_COLOR);
          cv::Mat gt = cv::imread(gt_dir + line, CV_LOAD_IMAGE_GRAYSCALE);
          ValidateExample(image, trimap, gt);
          patterns.push_back(new PatternData(line, image, trimap));
          labels.push_back(new LabelData(line, gt));
      }
  }
  main_file.close();

  examples=(EXAMPLE *)my_malloc(sizeof(EXAMPLE)*n);
  for (size_t i = 0; i < n; ++i) {
      examples[i].x = MakePattern(patterns[i]);
      examples[i].y = MakeLabel(labels[i]);
  }

  std::cout << " (" << n << " examples)... ";

  sample.n=n;
  sample.examples=examples;
  return(sample);
}

void        init_struct_model(SAMPLE sample, STRUCTMODEL *sm, 
			      STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, 
			      KERNEL_PARM *kparm)
{
  /* Initialize structmodel sm. The weight vector w does not need to be
     initialized, but you need to provide the maximum size of the
     feature space in sizePsi. This is the maximum number of different
     weights that can be learned. Later, the weight vector w will
     contain the learned weights for the model. */
    sm->data = new ModelData;

    sm->sizePsi=data(sm)->NumFeatures(); /* replace by appropriate number of features */
}

CONSTSET    init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm)
{
  /* Initializes the optimization problem. Typically, you do not need
     to change this function, since you want to start with an empty
     set of constraints. However, if for example you have constraints
     that certain weights need to be positive, you might put that in
     here. The constraints are represented as lhs[i]*w >= rhs[i]. lhs
     is an array of feature vectors, rhs is an array of doubles. m is
     the number of constraints. The function returns the initial
     set of constraints. */
  CONSTSET c;
  long     sizePsi=sm->sizePsi;
  long     i;
  WORD     words[2];

  if(1) { /* normal case: start with empty set of constraints */
    c.lhs=NULL;
    c.rhs=NULL;
    c.m=0;
  }
#if 0
  else { /* add constraints so that all learned weights are
            positive. WARNING: Currently, they are positive only up to
            precision epsilon set by -e. */
    c.lhs=my_malloc(sizeof(DOC *)*sizePsi);
    c.rhs=my_malloc(sizeof(double)*sizePsi);
    for(i=0; i<sizePsi; i++) {
      words[0].wnum=i+1;
      words[0].weight=1.0;
      words[1].wnum=0;
      /* the following slackid is a hack. we will run into problems,
         if we have move than 1000000 slack sets (ie examples) */
      c.lhs[i]=create_example(i,0,1000000+i,1,create_svector(words,"",1.0));
      c.rhs[i]=0.0;
    }
  }
#endif
  return(c);
}

LABEL       classify_struct_example(PATTERN x, STRUCTMODEL *sm, 
				    STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label yhat for pattern x that scores the highest
     according to the linear evaluation function in sm, especially the
     weights sm.w. The returned label is taken as the prediction of sm
     for the pattern x. The weights correspond to the features defined
     by psi() and range from index 1 to index sm->sizePsi. If the
     function cannot find a label, it shall return an empty label as
     recognized by the function empty_label(y). */
    LABEL y;

    CRF crf(0, 0);
    data(sm)->InitializeCRF(crf, *data(x));
    size_t feature_base = 1;
    for (auto fgp : data(sm)->m_features) {
        fgp->AddToCRF(crf, *data(x), sm->w + feature_base );
        feature_base += fgp->NumFeatures();
    }
    crf.Solve();
    y.data = data(sm)->ExtractLabel(crf, *data(x));

    return(y);
}

LABEL       find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label ybar for pattern x that that is responsible for
     the most violated constraint for the slack rescaling
     formulation. For linear slack variables, this is that label ybar
     that maximizes

            argmax_{ybar} loss(y,ybar)*(1-psi(x,y)+psi(x,ybar)) 

     Note that ybar may be equal to y (i.e. the max is 0), which is
     different from the algorithms described in
     [Tschantaridis/05]. Note that this argmax has to take into
     account the scoring function in sm, especially the weights sm.w,
     as well as the loss function, and whether linear or quadratic
     slacks are used. The weights in sm.w correspond to the features
     defined by psi() and range from index 1 to index
     sm->sizePsi. Most simple is the case of the zero/one loss
     function. For the zero/one loss, this function should return the
     highest scoring label ybar (which may be equal to the correct
     label y), or the second highest scoring label ybar, if
     Psi(x,ybar)>Psi(x,y)-1. If the function cannot find a label, it
     shall return an empty label as recognized by the function
     empty_label(y). */
  LABEL ybar;

  /* insert your code for computing the label ybar here */

  ASSERT(false /* Unimplemented! */);

  return(ybar);
}

LABEL       find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, 
						     STRUCTMODEL *sm, 
						     STRUCT_LEARN_PARM *sparm)
{
  /* Finds the label ybar for pattern x that that is responsible for
     the most violated constraint for the margin rescaling
     formulation. For linear slack variables, this is that label ybar
     that maximizes

            argmax_{ybar} loss(y,ybar)+psi(x,ybar)

     Note that ybar may be equal to y (i.e. the max is 0), which is
     different from the algorithms described in
     [Tschantaridis/05]. Note that this argmax has to take into
     account the scoring function in sm, especially the weights sm.w,
     as well as the loss function, and whether linear or quadratic
     slacks are used. The weights in sm.w correspond to the features
     defined by psi() and range from index 1 to index
     sm->sizePsi. Most simple is the case of the zero/one loss
     function. For the zero/one loss, this function should return the
     highest scoring label ybar (which may be equal to the correct
     label y), or the second highest scoring label ybar, if
     Psi(x,ybar)>Psi(x,y)-1. If the function cannot find a label, it
     shall return an empty label as recognized by the function
     empty_label(y). */
    LABEL ybar;

    /* insert your code for computing the label ybar here */
    CRF crf(0, 0);
    data(sm)->InitializeCRF(crf, *data(x));
    size_t feature_base = 1;
    for (auto fgp : data(sm)->m_features) {
        fgp->AddToCRF(crf, *data(x), sm->w + feature_base );
        feature_base += fgp->NumFeatures();
    }
    data(sm)->AddLossToCRF(crf, *data(x), *data(y));
    crf.Solve();
    ybar.data = data(sm)->ExtractLabel(crf, *data(x));


    return(ybar);
}

int         empty_label(LABEL y)
{
  /* Returns true, if y is an empty label. An empty label might be
     returned by find_most_violated_constraint_???(x, y, sm) if there
     is no incorrect label that can be found for x, or if it is unable
     to label x at all */
  return(0);
}

SVECTOR     *psi(PATTERN x, LABEL y, STRUCTMODEL *sm,
		 STRUCT_LEARN_PARM *sparm)
{
  /* Returns a feature vector describing the match between pattern x
     and label y. The feature vector is returned as a list of
     SVECTOR's. Each SVECTOR is in a sparse representation of pairs
     <featurenumber:featurevalue>, where the last pair has
     featurenumber 0 as a terminator. Featurenumbers start with 1 and
     end with sizePsi. Featuresnumbers that are not specified default
     to value 0. As mentioned before, psi() actually returns a list of
     SVECTOR's. Each SVECTOR has a field 'factor' and 'next'. 'next'
     specifies the next element in the list, terminated by a NULL
     pointer. The list can be though of as a linear combination of
     vectors, where each vector is weighted by its 'factor'. This
     linear combination of feature vectors is multiplied with the
     learned (kernelized) weight vector to score label y for pattern
     x. Without kernels, there will be one weight in sm.w for each
     feature. Note that psi has to match
     find_most_violated_constraint_???(x, y, sm) and vice versa. In
     particular, find_most_violated_constraint_???(x, y, sm) finds
     that ybar!=y that maximizes psi(x,ybar,sm)*sm.w (where * is the
     inner vector product) and the appropriate function of the
     loss + margin/slack rescaling method. See that paper for details. */
    SVECTOR *fvec=NULL;

    /* insert code for computing the feature vector for x and y here */

    std::vector<WORD> words;
    FNUM fnum = 1;
    WORD w;
    for (auto fgp : data(sm)->m_features) {
        std::vector<FVAL> values = fgp->Psi(*data(x), *data(y));
        ASSERT(values.size() == fgp->NumFeatures());
        for (FVAL v : values) {
            w.wnum = fnum++;
            w.weight = v;
            words.push_back(w);
        }
    }
    w.wnum = 0;
    words.push_back(w);

    fvec = create_svector(words.data(), nullptr, 1.0);

    return(fvec);
}

double      loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm)
{
  /* loss for correct label y and predicted label ybar. The loss for
     y==ybar has to be zero. sparm->loss_function is set with the -l option. */
  if(sparm->loss_function == 0) { /* type 0 loss: 0/1 loss */
                                  /* return 0, if y==ybar. return 1 else */
      return 1.0 - (double)(*data(y) == *data(ybar));
  }
  else {
    /* Put your code for different loss functions here. But then
       find_most_violated_constraint_???(x, y, sm) has to return the
       highest scoring label with the largest loss. */
      return data(y)->Loss(*data(ybar));
  }
}

int         finalize_iteration(double ceps, int cached_constraint,
			       SAMPLE sample, STRUCTMODEL *sm,
			       CONSTSET cset, double *alpha, 
			       STRUCT_LEARN_PARM *sparm)
{
  /* This function is called just before the end of each cutting plane iteration. ceps is the amount by which the most violated constraint found in the current iteration was violated. cached_constraint is true if the added constraint was constructed from the cache. If the return value is FALSE, then the algorithm is allowed to terminate. If it is TRUE, the algorithm will keep iterating even if the desired precision sparm->epsilon is already reached. */
    std::cout << "w = ";
    for (int i = 1; i <= data(sm)->NumFeatures(); ++i) {
        std::cout << sm->w[i] << ", ";
    }
    std::cout << "\n";
  return(0);
}

void        print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm,
					CONSTSET cset, double *alpha, 
					STRUCT_LEARN_PARM *sparm)
{
  /* This function is called after training and allows final touches to
     the model sm. But primarly it allows computing and printing any
     kind of statistic (e.g. training error) you might want. */
}

void        print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm,
				       STRUCT_LEARN_PARM *sparm, 
				       STRUCT_TEST_STATS *teststats)
{
  /* This function is called after making all test predictions in
     svm_struct_classify and allows computing and printing any kind of
     evaluation (e.g. precision/recall) you might want. You can use
     the function eval_prediction to accumulate the necessary
     statistics for each prediction. */
}

void        eval_prediction(long exnum, EXAMPLE ex, LABEL ypred, 
			    STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, 
			    STRUCT_TEST_STATS *teststats)
{
  /* This function allows you to accumlate statistic for how well the
     predicition matches the labeled example. It is called from
     svm_struct_classify. See also the function
     print_struct_testing_stats. */
  if(exnum == 0) { /* this is the first time the function is
		      called. So initialize the teststats */
  }
}

void        write_struct_model(char *file, STRUCTMODEL *sm, 
			       STRUCT_LEARN_PARM *sparm)
{
  /* Writes structural model sm to file file. */
    MODEL *model = sm->svm_model;
    SVECTOR *v;
    long j,i,sv_num;

    std::ofstream ofs(file, std::ios_base::trunc | std::ios_base::out);
    boost::archive::text_oarchive ar(ofs);



    std::string version = std::string(INST_VERSION);
    ar & version;
    ar & sparm->loss_function;
    ar & model->kernel_parm.kernel_type;
    ar & model->kernel_parm.poly_degree;
    ar & model->kernel_parm.rbf_gamma;
    ar & model->kernel_parm.coef_lin;
    ar & model->kernel_parm.coef_const;
    ar & model->kernel_parm.custom;
    ar & model->totwords;
    ar & model->totdoc;

    sv_num=1;
    for(i=1;i<model->sv_num;i++) {
    for(v=model->supvec[i]->fvec;v;v=v->next) 
      sv_num++;
    }
    ar & sv_num;
    ar & model->b;

    ar & *data(sm);

    for(i=1;i<model->sv_num;i++) {
    for(v=model->supvec[i]->fvec;v;v=v->next) {
        double factor = (model->alpha[i]*v->factor);
        ar & factor;
        ar & v->kernel_id;
        size_t num_words = 0;
        for (j=0; (v->words[j]).wnum; j++) num_words++;
        ar & num_words;
        for (j=0; (v->words[j]).wnum; j++) {
            ar & (v->words[j]).wnum;
            ar & (v->words[j]).weight;
        }
        ASSERT(!v->userdefined);
    /* NOTE: this could be made more efficient by summing the
       alpha's of identical vectors before writing them to the
       file. */
    }
    }
    ofs.close();
}

STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads structural model sm from file file. This function is used
     only in the prediction module, not in the learning module. */
    STRUCTMODEL sm;
    std::ifstream ifs(file);
    boost::archive::text_iarchive ar(ifs);

    sm.svm_model = (MODEL*)my_malloc(sizeof(MODEL));
    MODEL* model = sm.svm_model;

    std::string inst_version;
    ar & inst_version;
    ASSERT(inst_version == std::string(INST_VERSION));
    ar & sparm->loss_function;
    ar & model->kernel_parm.kernel_type;
    ar & model->kernel_parm.poly_degree;
    ar & model->kernel_parm.rbf_gamma;
    ar & model->kernel_parm.coef_lin;
    ar & model->kernel_parm.coef_const;
    ar & model->kernel_parm.custom;
    ar & model->totwords;
    ar & model->totdoc;

    ar & model->sv_num;
    ar & model->b;

    model->supvec = (DOC **)my_malloc(sizeof(DOC *)*model->sv_num);
    model->alpha = (double *)my_malloc(sizeof(double)*model->sv_num);
    model->index=NULL;
    model->lin_weights=NULL;


    sm.data = new ModelData;
    ar & *data(sm);

    for(int i = 1; i < model->sv_num; i++) {
        long kernel_id;
        size_t num_words;
        ar & model->alpha[i];
        ar & kernel_id;
        ar & num_words;
        std::vector<WORD> words;
        WORD w;
        for (size_t j = 0; j < num_words; j++) {
            ar & w.wnum;
            ar & w.weight;
            words.push_back(w);
        }
        w.wnum = 0;
        words.push_back(w);
        model->supvec[i] = create_example(-1, 0, 0, 0.0, create_svector(words.data(), nullptr, 1.0));
        model->supvec[i]->fvec->kernel_id = kernel_id;
    }
    ifs.close();
    return sm;
}

void        write_label(FILE *fp, LABEL y)
{
  /* Writes label y to file handle fp. */
    std::ostringstream os;
    boost::archive::text_oarchive oa(os);
    oa << *data(y);
    fwrite(os.str().c_str(), sizeof(char), os.str().size()+1, fp);
    ShowImage(data(y)->m_gt);
} 

void        free_pattern(PATTERN x) {
    delete data(x);
}

void        free_label(LABEL y) {
    delete data(y);
}

void        free_struct_model(STRUCTMODEL sm) 
{
  /* Frees the memory of model. */
  /* if(sm.w) free(sm.w); */ /* this is free'd in free_model */
  if(sm.svm_model) free_model(sm.svm_model,1);
  /* add free calls for user defined data here */
  delete data(sm);
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
  printf("         --* string  -> custom parameters that can be adapted for struct\n");
  printf("                        learning. The * can be replaced by any character\n");
  printf("                        and there can be multiple options starting with --.\n");
}

void         parse_struct_parameters(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- */
  int i;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2]) 
      { 
      case 'a': i++; /* strcpy(learn_parm->alphafile,argv[i]); */ break;
      case 'e': i++; /* sparm->epsilon=atof(sparm->custom_argv[i]); */ break;
      case 'k': i++; /* sparm->newconstretrain=atol(sparm->custom_argv[i]); */ break;
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}

void        print_struct_help_classify()
{
  /* Prints a help text that is appended to the common help text of
     svm_struct_classify. */
  printf("         --* string -> custom parameters that can be adapted for struct\n");
  printf("                       learning. The * can be replaced by any character\n");
  printf("                       and there can be multiple options starting with --.\n");
}

void         parse_struct_parameters_classify(STRUCT_LEARN_PARM *sparm)
{
  /* Parses the command line parameters that start with -- for the
     classification module */
  int i;

  for(i=0;(i<sparm->custom_argc) && ((sparm->custom_argv[i])[0] == '-');i++) {
    switch ((sparm->custom_argv[i])[2]) 
      { 
      /* case 'x': i++; strcpy(xvalue,sparm->custom_argv[i]); break; */
      default: printf("\nUnrecognized option %s!\n\n",sparm->custom_argv[i]);
	       exit(0);
      }
  }
}

