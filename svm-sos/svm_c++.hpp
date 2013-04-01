#ifndef _SVM_CXX_HPP_
#define _SVM_CXX_HPP_

extern "C" {
#include "svm_light/svm_common.h"
#include "svm_struct/svm_struct_common.h"
}
#include <string>

class PatternData {
    public:
    PatternData() = default;
    PatternData(const std::string& name) : m_name(name) { }
    virtual ~PatternData() { }

    const std::string& Name() const { return m_name; }

    protected:
    std::string m_name;
};

class LabelData {
    public:
        LabelData() = default;
        explicit LabelData(const std::string& name) : m_name(name) { }
        virtual ~LabelData() { }

        const std::string& Name() const { return m_name; }

    protected:
        std::string m_name;
};

class SVM_App_Base {
    public:
        SVM_App_Base() { }
        virtual ~SVM_App_Base() { }

        // Forwarding functions for api
        virtual void svm_struct_learn_api_exit() = 0;
        virtual void svm_struct_classify_api_exit() = 0;
        virtual SAMPLE read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm) = 0;
        virtual void init_struct_model(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, KERNEL_PARM *kparm) = 0;
        virtual CONSTSET init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) = 0;
        virtual LABEL classify_struct_example(PATTERN x, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) = 0;
        virtual LABEL find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) = 0;
        virtual LABEL find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) = 0;
        virtual int empty_label(LABEL y) = 0;
        virtual SVECTOR* psi(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) = 0;
        virtual double loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm) = 0;
        virtual int finalize_iteration(double ceps, int cached_constraint, SAMPLE sample, STRUCTMODEL *sm, CONSTSET cset, double *alpha, STRUCT_LEARN_PARM *sparm) = 0;
        virtual void print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm, CONSTSET cset, double *alpha, STRUCT_LEARN_PARM *sparm) = 0;
        virtual void print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, STRUCT_TEST_STATS *teststats) = 0;
        virtual void eval_prediction(long exnum, EXAMPLE ex, LABEL ypred, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, STRUCT_TEST_STATS *teststats) = 0;
        virtual void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) = 0;
        virtual STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm) = 0;
        virtual void write_label(FILE* fp, LABEL y) = 0;
};

template <class T>
struct AppTraits { };

template <class Derived>
class SVM_App : public SVM_App_Base {
    public:
        typedef typename AppTraits<Derived>::PatternData DPatternData;
        typedef typename AppTraits<Derived>::LabelData DLabelData;

        SVM_App(Derived* d) : m_derived(d) { }
        virtual ~SVM_App() { }

    private:
        Derived* m_derived;
        static DPatternData* Downcast(PatternData* p) { return static_cast<DPatternData*>(p); }
        static DLabelData* Downcast(LabelData* p) { return static_cast<DLabelData*>(p); }

        // Forwarding functions for api
        virtual void svm_struct_learn_api_exit() override;
        virtual void svm_struct_classify_api_exit() override;
        virtual SAMPLE read_struct_examples(char *file, STRUCT_LEARN_PARM *sparm) override;
        virtual void init_struct_model(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, LEARN_PARM *lparm, KERNEL_PARM *kparm) override;
        virtual CONSTSET init_struct_constraints(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) override;
        virtual LABEL classify_struct_example(PATTERN x, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) override;
        virtual LABEL find_most_violated_constraint_slackrescaling(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) override;
        virtual LABEL find_most_violated_constraint_marginrescaling(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) override;
        virtual int empty_label(LABEL y) override;
        virtual SVECTOR* psi(PATTERN x, LABEL y, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) override;
        virtual double loss(LABEL y, LABEL ybar, STRUCT_LEARN_PARM *sparm) override;
        virtual int finalize_iteration(double ceps, int cached_constraint, SAMPLE sample, STRUCTMODEL *sm, CONSTSET cset, double *alpha, STRUCT_LEARN_PARM *sparm) override;
        virtual void print_struct_learning_stats(SAMPLE sample, STRUCTMODEL *sm, CONSTSET cset, double *alpha, STRUCT_LEARN_PARM *sparm) override;
        virtual void print_struct_testing_stats(SAMPLE sample, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, STRUCT_TEST_STATS *teststats) override;
        virtual void eval_prediction(long exnum, EXAMPLE ex, LABEL ypred, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm, STRUCT_TEST_STATS *teststats) override;
        virtual void write_struct_model(char *file, STRUCTMODEL *sm, STRUCT_LEARN_PARM *sparm) override;
        virtual STRUCTMODEL read_struct_model(char *file, STRUCT_LEARN_PARM *sparm) override;
        virtual void write_label(FILE* fp, LABEL y) override;
};

extern SVM_App_Base* g_application;

#endif

