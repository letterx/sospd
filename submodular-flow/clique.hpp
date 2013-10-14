#ifndef _CLIQUE_HPP_
#define _CLIQUE_HPP_

#include "sos-common.hpp"
#include "submodular-flow.hpp"
#include "submodular-ibfs.hpp"

typedef int NodeId;
typedef size_t Label;

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
        REAL m_f_min, m_f_max;

    protected:
        std::vector<NodeId> m_nodes;

        // Remove move and copy operators to prevent slicing of base classes
        Clique(Clique&&) = delete;
        Clique& operator=(Clique&&) = delete;
        Clique(const Clique&) = delete;
        Clique& operator=(const Clique&) = delete;
};

class PottsClique : public Clique {
    public:

        PottsClique(const std::vector<NodeId>& nodes, REAL same_cost, REAL diff_cost)
            : Clique(nodes),
            m_same_cost(same_cost),
            m_diff_cost(diff_cost)
        { }

        virtual REAL Energy(const std::vector<Label>& labels) const override {
            const Label l = labels[0];
            for (Label l2 : labels)
                if (l2 != l)
                    return m_diff_cost;
            return m_same_cost;
        }
    private:
        REAL m_same_cost;
        REAL m_diff_cost;
};

class SeparableClique : public Clique {
    public:
        typedef uint32_t Assgn;
        typedef std::vector<std::vector<REAL>> EnergyTable;

        SeparableClique(const std::vector<NodeId>& nodes, const EnergyTable& energy_table)
            : Clique(nodes),
            m_energy_table(energy_table) { }

        virtual REAL Energy(const std::vector<Label>& labels) const override {
            ASSERT(labels.size() == this->m_nodes.size());
            const Label num_labels = m_energy_table.size();
            std::vector<Assgn> per_label(num_labels, 0);
            for (size_t i = 0; i < labels.size(); ++i)
                per_label[labels[i]] |= 1 << i;
            REAL e = 0;
            for (Label l = 0; l < num_labels; ++l)
                e += m_energy_table[l][per_label[l]];
            return e;
        }
    private:
        EnergyTable m_energy_table;
};

#endif
