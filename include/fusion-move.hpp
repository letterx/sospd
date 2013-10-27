#ifndef _FUSION_MOVE_HPP_
#define _FUSION_MOVE_HPP_

/*
 * fusion-move.hpp
 *
 * Copyright 2012 Alexander Fix
 * See LICENSE.txt for license information
 *
 * Computes a fusion move between the current and proposed image.
 *
 * A fusion move takes two images (current and proposed) and tries to perform
 * the optimal move where each pixel is allowed to either stay at its current
 * value, or switch to its label in the proposed image. This is a 
 * generalization of alpha-expansion, where in alpha-expansion each pixel is 
 * allowed to either stay the same, or change to a fixed value alpha. That is,
 * alpha expansion is a fusion move where the proposed image is just the flat
 * image with value alpha at all pixels.
 */

#include <iostream>
#include <sstream>
#include <boost/foreach.hpp>
#include <functional>
#include "higher-order-energy.hpp"
#include "clique.hpp"
#include "QPBO.h"

template <int MaxDegree>
class FusionMove {
    public:
        typedef MultilabelEnergy::NodeId NodeId;
        typedef MultilabelEnergy::Label Label;
        typedef std::vector<Label> LabelVec;
        typedef std::function<void(int, const LabelVec&, LabelVec&)> ProposalCallback;
        FusionMove(const MultilabelEnergy* energy, const ProposalCallback& pc)
            : m_energy(energy), m_pc(pc), m_labels(energy->NumNodes(), 0), m_iter(0) { }
        FusionMove(const MultilabelEnergy* energy, const ProposalCallback& pc, const LabelVec& current)
            : m_energy(energy), m_pc(pc), m_labels(current), m_iter(0) { }

        void Solve(int niters);
        Label GetLabel(NodeId i) const { return m_labels[i]; }

    protected:
        void SetupFusionEnergy(const LabelVec& proposed,
                HigherOrderEnergy<REAL, MaxDegree>& hoe) const;
        void GetFusedImage(const LabelVec& proposed, QPBO<REAL>& qr);
        void FusionStep();
    
        const MultilabelEnergy* m_energy;
        ProposalCallback m_pc;
        LabelVec m_labels;
        int m_iter;
};

template <int MaxDegree>
void FusionMove<MaxDegree>::Solve(int niters) {
    for (int i = 0; i < niters; ++i)
        FusionStep();
}

template <int MaxDegree>
void FusionMove<MaxDegree>::FusionStep() {
    HigherOrderEnergy<REAL, MaxDegree> hoe;
    QPBO<REAL> qr(m_labels.size(), 0);
    LabelVec proposed(m_labels.size());
    m_pc(m_iter, m_labels, proposed);
    SetupFusionEnergy(proposed, hoe);
    hoe.ToQuadratic(qr);
    qr.Solve();
    qr.ComputeWeakPersistencies();
    GetFusedImage(proposed, qr);
    m_iter++;
}

template <int MaxDegree>
void FusionMove<MaxDegree>::GetFusedImage(const LabelVec& proposed, QPBO<REAL>& qr) {
    for (size_t i = 0; i < m_labels.size(); ++i) {
        int label = qr.GetLabel(i);
        if (label == 1) {
            m_labels[i] = proposed[i];
        }
    }
}

template <int MaxDegree>
void FusionMove<MaxDegree>::SetupFusionEnergy(const LabelVec& proposed,
        HigherOrderEnergy<REAL, MaxDegree>& hoe) const {
    hoe.AddVars(m_energy->NumNodes());
    for (NodeId i = 0; i < m_energy->NumNodes(); ++i)
        hoe.AddUnaryTerm(i, m_energy->Unary(i, m_labels[i]), m_energy->Unary(i, proposed[i]));

    std::vector<REAL> energy_table;
    for (const auto& cp : m_energy->Cliques()) {
        const Clique& c = *cp;
        NodeId size = c.Size();
        ASSERT(size > 1);

        uint32_t numAssignments = 1 << size;
        energy_table.resize(numAssignments);
        
        // For each boolean assignment, get the clique energy at the 
        // corresponding labeling
        Label cliqueLabels[size];
        for (uint32_t assignment = 0; assignment < numAssignments; ++assignment) {
            for (NodeId i = 0; i < size; ++i) {
                if (assignment & (1 << i)) { 
                    cliqueLabels[i] = proposed[c.Nodes()[i]];
                } else {
                    cliqueLabels[i] = m_labels[c.Nodes()[i]];
                }
            }
            energy_table[assignment] = c.Energy(cliqueLabels);
        }
        std::vector<NodeId> nodes(c.Nodes(), c.Nodes() + c.Size());
        hoe.AddClique(nodes, energy_table);
    }
}

#endif
