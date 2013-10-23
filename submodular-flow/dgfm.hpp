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
#include <functional>

class DualGuidedFusionMove {
    public:
        typedef MultilabelEnergy::NodeId NodeId;
        typedef MultilabelEnergy::Label Label;
        typedef MultilabelEnergy::CliquePtr CliquePtr;
        typedef std::vector<std::vector<REAL> > Dual;
        typedef std::vector<std::vector<std::pair<size_t, size_t> > > NodeCliqueList;
        typedef std::function<void(int, const std::vector<Label>&, std::vector<Label>&)> ProposalCallback;

        DualGuidedFusionMove() = delete;
        explicit DualGuidedFusionMove(const MultilabelEnergy* energy);

        void Solve(int niters = std::numeric_limits<int>::max());
        int GetLabel(NodeId i) const;

        void SetExpansionSubmodular(bool b) { m_expansion_submodular = b; }
        void SetProposalCallback(const ProposalCallback& pc) { m_pc = pc; }
        // Set the proposal method to regular alpha-expansion
        void SetAlphaExpansion() { 
            m_pc = [&](int, const std::vector<Label>&, std::vector<Label>&) {
                AlphaProposal();
            };
        }
        // Set the proposal method to best-height alpha-expansion
        void SetHeightAlphaExpansion() { 
            m_pc = [&](int, const std::vector<Label>&, std::vector<Label>&) {
                HeightAlphaProposal();
            };
        }

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
        REAL& Height(NodeId i, Label l) { return m_heights[i*m_num_labels+l]; }

        // Move Proposals
        void HeightAlphaProposal();
        void AlphaProposal();

        const MultilabelEnergy* m_energy;
        SubmodularIBFS m_ibfs;
        const size_t m_num_labels;
        std::vector<Label> m_labels;
        std::vector<Label> m_fusion_labels;
        NodeCliqueList m_node_clique_list;
        std::vector<Dual> m_dual;
        std::vector<REAL> m_heights;
        bool m_expansion_submodular;
        int m_iter;
        ProposalCallback m_pc;
};

#endif
