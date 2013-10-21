#ifndef _SPD2_HPP_
#define _SPD2_HPP_

#define PROGRESS_DISPLAY
//#define CHECK_INVARIANTS

#include "energy-common.hpp"
#include "clique.hpp"
#include "submodular-flow.hpp"
#include "submodular-ibfs.hpp"
#include <vector>

class SubmodularPrimalDual2 {
    public:
        typedef MultilabelEnergy::NodeId NodeId;
        typedef MultilabelEnergy::Label Label;
        typedef MultilabelEnergy::CliquePtr CliquePtr;
        typedef std::vector<std::vector<REAL> > Dual;
        typedef std::vector<std::vector<std::pair<size_t, size_t> > > NodeCliqueList;

        SubmodularPrimalDual2() = delete;
        explicit SubmodularPrimalDual2(const MultilabelEnergy* energy);
        
        void Solve();
        int GetLabel(NodeId i) const;

    protected:
        REAL ComputeHeight(NodeId, Label);
        REAL ComputeHeightDiff(NodeId i, Label l1, Label l2) const;
        void SetupGraph(SubmodularIBFS& crf);
        void SetupAlphaEnergy(Label alpha, SubmodularIBFS& crf);
        void InitialLabeling();
        void InitialDual();
        void InitialNodeCliqueList();
        void PreEditDual(Label alpha);
        bool UpdatePrimalDual(Label alpha, SubmodularIBFS& crf);
        void PostEditDual();
        void DualFit();
        bool CheckHeightInvariant();
        bool CheckLabelInvariant();
        bool CheckDualBoundInvariant();
        bool CheckActiveInvariant();

        const MultilabelEnergy* m_energy;
        Label m_num_labels;
        std::vector<Label> m_labels;
        NodeCliqueList m_node_clique_list;
        std::vector<Dual> m_dual;
};

#endif
