#ifndef _SPD2_HPP_
#define _SPD2_HPP_

#define PROGRESS_DISPLAY
//#define DEBUG

#include "sos-common.hpp"
#include "clique.hpp"
#include "submodular-flow.hpp"
#include "submodular-ibfs.hpp"
#include <vector>

class SubmodularPrimalDual2 {
    public:
        typedef size_t NodeId;
        typedef size_t Label;
        typedef std::shared_ptr<Clique> CliquePtr;
        typedef std::vector<REAL> UnaryCost;
        typedef std::vector<std::vector<REAL> > Dual;
        typedef std::vector<std::vector<std::pair<size_t, size_t> > > NodeCliqueList;

        SubmodularPrimalDual2() = delete;
        explicit SubmodularPrimalDual2(Label max_label);
        
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
        void SetupAlphaEnergy(Label alpha, SubmodularIBFS& crf);
        void InitialLabeling();
        void InitialDual();
        void InitialNodeCliqueList();
        void PreEditDual(Label alpha);
        bool UpdatePrimalDual(Label alpha);
        void PostEditDual();
        void DualFit();
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
        REAL m_mu;
        REAL m_rho;
};

#endif
