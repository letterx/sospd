#ifndef _DGFM_HPP_
#define _DGFM_HPP_

#define PROGRESS_DISPLAY
//#define CHECK_INVARIANTS

#include "energy-common.hpp"
#include "clique.hpp"
#include "submodular-flow.hpp"
#include "submodular-ibfs.hpp"
#include "submodular-functions.hpp"
#include <vector>

class DualGuidedFusionMove {
    public:
        typedef MultilabelEnergy::NodeId NodeId;
        typedef MultilabelEnergy::Label Label;
        typedef MultilabelEnergy::CliquePtr CliquePtr;
        typedef std::vector<std::vector<REAL> > Dual;
        typedef std::vector<std::vector<std::pair<size_t, size_t> > > NodeCliqueList;

        DualGuidedFusionMove() = delete;
        explicit DualGuidedFusionMove(const MultilabelEnergy* energy);

        void Solve(int niters = std::numeric_limits<int>::max());
        int GetLabel(NodeId i) const;

        void SetExpansionSubmodular(bool b) { m_expansion_submodular = b; }

    protected:
        REAL ComputeHeight(NodeId, Label);
        REAL ComputeHeightDiff(NodeId i, Label l1, Label l2) const;
        void SetupGraph(SubmodularIBFS& crf);
        void SetupAlphaEnergy(SubmodularIBFS& crf);
        void InitialLabeling();
        void InitialDual();
        void InitialNodeCliqueList();
        bool InitialFusionLabeling();
        void PreEditDual(SubmodularIBFS& crf);
        bool UpdatePrimalDual(SubmodularIBFS& crf);
        void PostEditDual();
        void DualFit();
        bool CheckHeightInvariant();
        bool CheckLabelInvariant();
        bool CheckDualBoundInvariant();
        bool CheckActiveInvariant();

        const MultilabelEnergy* m_energy;
        SubmodularIBFS m_ibfs;
        const size_t m_num_labels;
        std::vector<Label> m_labels;
        std::vector<Label> m_fusion_labels;
        NodeCliqueList m_node_clique_list;
        std::vector<Dual> m_dual;
        bool m_expansion_submodular;
        int m_iter;
};

#endif
