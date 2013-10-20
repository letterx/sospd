#ifndef _SPD1_HPP_
#define _SPD1_HPP_

#define PROGRESS_DISPLAY
//#define DEBUG

#include "energy-common.hpp"
#include "clique.hpp"
#include "ibfs.h"
#include <vector>

class SubmodularPrimalDual1 {
    public:
        typedef size_t NodeId;
        typedef size_t Label;
        typedef double flowtype;
        typedef std::pair<int, int> edgeInfo;
        typedef std::pair<edgeInfo, flowtype> flowInfo;
        typedef std::vector<flowInfo> flowList;
        typedef std::pair<flowtype, flowtype> STFlowInfo;
        typedef std::vector<STFlowInfo> STFlowList;
        typedef std::shared_ptr<Clique> CliquePtr;
        typedef std::vector<REAL> UnaryCost;
        typedef std::vector<std::vector<REAL> > Dual;
        typedef std::vector<std::vector<std::pair<size_t, size_t> > > NodeCliqueList;

        SubmodularPrimalDual1() = delete;
        explicit SubmodularPrimalDual1(Label max_label);
        
        void SetMu(double mu);
        double GetMu();
        void ComputeRho();
        double GetRho();

        NodeId AddNode(int i = 1);
        int GetLabel(NodeId i) const;

        void AddConstantTerm(REAL c);
        void AddUnaryTerm(NodeId i, const std::vector<REAL>& coeffs);
        void AddClique(const CliquePtr& cp);

        void Solve();

        REAL ComputeEnergy() const;
        REAL ComputeEnergy(const std::vector<Label>& labels) const;

    protected:
        REAL ComputeHeight(NodeId, Label);
        void SetupAlphaEnergy(Label alpha, IBFSGraph<flowtype, flowtype, flowtype>& crf);
        void InitialLabeling();
        void InitialDual();
        void InitialNodeCliqueList();
        bool UpdatePrimalDual(Label alpha);
        void PostEditDual();
        bool CheckHeightInvariant();
        bool CheckLabelInvariant();
        bool CheckDualBoundInvariant();
        bool CheckActiveInvariant();
        bool CheckZeroSumInvariant();
        const double EPS = 1e-7;
        const size_t m_num_labels;
        size_t m_num_cliques;
        REAL m_constant_term;
        std::vector<CliquePtr> m_cliques;
        std::vector<UnaryCost> m_unary_cost;
        std::vector<Label> m_labels;
        NodeCliqueList m_node_clique_list;
        std::vector<Dual> m_dual;
        REAL m_rho;
};

#endif
