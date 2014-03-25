#ifndef _SOSPD_HPP_
#define _SOSPD_HPP_

/** \file sospd.hpp
 * Sum-of-submodular Primal Dual algorithm for multilabel problems
 */

//#define PROGRESS_DISPLAY
//#define CHECK_INVARIANTS

#include "energy-common.hpp"
#include <vector>
#include <functional>

#include "clique.hpp"
#include "submodular-ibfs.hpp"
#include "submodular-functions.hpp"


/** Optimizer using Sum-of-submodular Primal Dual algorithm. 
 *
 * Implements SoSPD algorithm from Fix, Wang, Zabih in CVPR 14.
 */
class SoSPD {
    public:
        typedef MultilabelEnergy::NodeId NodeId;
        typedef MultilabelEnergy::Label Label;

        /** Proposal callbacks take as input the iteration number and current
         * labeling (as a vector of labels) and write the next proposal to the
         * final parameter.
         */
        typedef std::function<
              void(int niter,
                   const std::vector<Label>& current,
                   std::vector<Label>& proposed)
            > ProposalCallback;

        /** Set up SoSPD to optimize a particular energy function
         *
         * \param energy Energy function to optimize.
         */
        explicit SoSPD(const MultilabelEnergy* energy);

        /** Run SoSPD algorithm either to completion, or for a number of steps.
         *
         * Each iteration has a single proposal (determined by 
         * SetProposalCallback), and solves a corresponding Sum-of-Submodular
         * flow problem. 
         *
         * Resulting labeling can be queried from GetLabel.
         *
         * \param niters Number of iterations
         */
        void Solve(int niters = std::numeric_limits<int>::max());

        /** Return label of a node i, returns -1 if Solve has not been called.*/
        int GetLabel(NodeId i) const;

        /** Enable optimization if energy known to be expansion submodular. */
        void SetExpansionSubmodular(bool b) { m_expansion_submodular = b; }

        /** Choose whether to use lower or upper bound in approximating function.
         */
        void SetLowerBound(bool b) { m_lower_bound = b; }

        /** Specify method for choosing proposals. */
        void SetProposalCallback(const ProposalCallback& pc) { m_pc = pc; }

        /** Set the proposal method to alpha-expansion 
         *
         * Alpha-expansion proposals simply cycle through the labels, proposing
         * a constant labeling (i.e., all "alpha") at each iteration.
         */
        void SetAlphaExpansion() { 
            m_pc = [&](int, const std::vector<Label>&, std::vector<Label>&) {
                AlphaProposal();
            };
        }

        /** Set the proposal method to best-height alpha-expansion
         *
         * Best-height alpha-expansion, instead of cycling through labels, 
         * chooses the single alpha with the biggest sum of differences in 
         * heights.
         */
        void SetHeightAlphaExpansion() { 
            m_pc = [&](int, const std::vector<Label>&, std::vector<Label>&) {
                HeightAlphaProposal();
            };
        }

        /** Return lower bound on optimum, determined by current dual */
        double LowerBound();

    private:
        typedef MultilabelEnergy::CliquePtr CliquePtr;
        typedef std::vector<std::vector<REAL> > Dual;
        typedef std::vector<std::pair<size_t, size_t>> NodeNeighborList;
        typedef std::vector<NodeNeighborList> NodeCliqueList;

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
        bool m_lower_bound;
        int m_iter;
        ProposalCallback m_pc;
};

#endif
