#ifndef _ALPHA_EXPANSION_HPP_
#define _ALPHA_EXPANSION_HPP_

#include "sos-common.hpp"
#include "submodular-flow.hpp"
#include <vector>

class MultiLabelCRF {
    public:
        typedef size_t Label;
        typedef int NodeId;
        struct Clique;
        typedef std::shared_ptr<Clique> CliquePtr;
        typedef std::vector<REAL> UnaryCost;

        MultiLabelCRF() = delete;
        explicit MultiLabelCRF(Label max_label);

        NodeId AddNode(int i = 1);
        int GetLabel(NodeId i) const;

        void AddConstantTerm(REAL c);
        void AddUnaryTerm(NodeId i, const std::vector<REAL>& coeffs);
        void AddClique(const CliquePtr& cp);

        void AlphaExpansion();
        void Solve();

        REAL ComputeEnergy() const;
        REAL ComputeEnergy(const std::vector<Label>& labels) const;

        class Clique {
            public:
            Clique(const std::vector<NodeId>& nodes)
                : m_nodes(nodes)
            { }
            virtual ~Clique() = default;

            // labels is a vector of length m_nodes.size() with the labels
            // of m_nodes. Returns the energy of the clique at that labeling
            virtual REAL Energy(const std::vector<Label>& labels) const = 0;

            const std::vector<NodeId>& Nodes() const { return m_nodes; }
            size_t Size() const { return m_nodes.size(); }

            protected:
            std::vector<NodeId> m_nodes;

            // Remove move and copy operators to prevent slicing of base classes
            Clique(Clique&&) = delete;
            Clique& operator=(Clique&&) = delete;
            Clique(const Clique&) = delete;
            Clique& operator=(const Clique&) = delete;
        };

    protected:
        void SetupAlphaEnergy(Label alpha, SubmodularFlow& crf) const;
        const size_t m_num_labels;
        REAL m_constant_term;
        std::vector<CliquePtr> m_cliques;
        std::vector<UnaryCost> m_unary_cost;
        std::vector<Label> m_labels;
};







#endif
