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
void SVM_App<Derived>::print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm,
				       STRUCT_LEARN_PARM *sparm, 
				       STRUCT_TEST_STATS *teststats)
{
  /* This function is called after making all test predictions in
     svm_struct_classify and allows computing and printing any kind of
     evaluation (e.g. precision/recall) you might want. You can use
     the function eval_prediction to accumulate the necessary
     statistics for each prediction. */
    if (m_derived->Params().stats_file != std::string())
        m_test_stats.Write(m_derived->Params().stats_file);
}

template <class Derived>
void SVM_App<Derived>::eval_prediction(long exnum, EXAMPLE ex, LABEL ypred, 
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

    double loss = m_derived->Loss(*Downcast(ex.y.data), *Downcast(ypred.data), sparm->loss_scale)*100.0 / sparm->loss_scale;
    m_test_stats.Add(TestStats::ImageStats(ex.x.data->Name(), loss, m_test_stats.LastTime()));

    m_derived->EvalPrediction(*Downcast(ex.x.data), *Downcast(ex.y.data), *Downcast(ypred.data));
}

template <class Derived>
void SVM_App<Derived>::write_struct_model(char *file, STRUCTMODEL *sm, 
			       STRUCT_LEARN_PARM *sparm)
{
  /* Writes structural model sm to file file. */
    MODEL *model = sm->svm_model;
    SVECTOR *v;
    long j,i,sv_num;

    m_test_stats.m_model_file = std::string(file);

    std::ofstream ofs(file, std::ios_base::trunc | std::ios_base::out);
    boost::archive::text_oarchive ar(ofs);



    std::string version = std::string(INST_VERSION);
    ar & version;
    ar & sparm->loss_function;
    ar & sparm->constraint_scale;
    ar & sparm->feature_scale;
    ar & sparm->loss_scale;
    ar & model->kernel_parm.kernel_type;
    ar & model->kernel_parm.poly_degree;
    ar & model->kernel_parm.rbf_gamma;
    ar & model->kernel_parm.coef_lin;
    ar & model->kernel_parm.coef_const;
    ar & model->kernel_parm.custom;
    ar & model->totwords;
    ar & model->totdoc;

    unsigned int param_version = m_derived->Params().Version();
    ar & param_version;
    m_derived->SerializeParams(ar, param_version);
    ar & m_test_stats;

    sv_num=1;
    for(i=1;i<model->sv_num;i++) {
    for(v=model->supvec[i]->fvec;v;v=v->next) 
      sv_num++;
    }
    ar & sv_num;
    ar & model->b;

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

template <class Derived>
STRUCTMODEL SVM_App<Derived>::read_struct_model(char *file, STRUCT_LEARN_PARM *sparm)
{
  /* Reads structural model sm from file file. This function is used
     only in the prediction module, not in the learning module. */
    STRUCTMODEL sm;
    std::ifstream ifs(file);
    boost::archive::text_iarchive ar(ifs);
    
    strcpy(sparm->model_file, file);

    sm.svm_model = (MODEL*)my_malloc(sizeof(MODEL));
    MODEL* model = sm.svm_model;

    std::string inst_version;
    ar & inst_version;
    ASSERT(inst_version == std::string(INST_VERSION));
    ar & sparm->loss_function;
    ar & sparm->constraint_scale;
    ar & sparm->feature_scale;
    ar & sparm->loss_scale;
    ar & model->kernel_parm.kernel_type;
    ar & model->kernel_parm.poly_degree;
    ar & model->kernel_parm.rbf_gamma;
    ar & model->kernel_parm.coef_lin;
    ar & model->kernel_parm.coef_const;
    ar & model->kernel_parm.custom;
    ar & model->totwords;
    ar & model->totdoc;

    unsigned int version;
    ar & version;
    m_derived->SerializeParams(ar, version);
    m_derived->InitFeatures(m_derived->Params());
    ar & m_test_stats;

    ar & model->sv_num;
    ar & model->b;

    model->supvec = (DOC **)my_malloc(sizeof(DOC *)*model->sv_num);
    model->alpha = (double *)my_malloc(sizeof(double)*model->sv_num);
    model->index=NULL;
    model->lin_weights=NULL;


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

