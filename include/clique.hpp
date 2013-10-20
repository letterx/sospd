#ifndef _CLIQUE_HPP_
#define _CLIQUE_HPP_

#include "energy-common.hpp"

typedef int NodeId;
typedef size_t Label;

class Clique {
    public:   
        Clique() { }
        virtual ~Clique() = default;

        // labels must be an array of length this->Size() with the labels
        // corresponding to the nodes.
        // Returns the energy of the clique at that labeling
        virtual REAL Energy(const Label* labels) const = 0;
        virtual const NodeId* Nodes() const = 0;
        virtual size_t Size() const = 0;
        virtual double Rho() const { return std::numeric_limits<double>::infinity(); }
        virtual REAL FMax() const { return std::numeric_limits<REAL>::max(); }

    private:
        // Remove move and copy operators to prevent slicing of base classes
        Clique(Clique&&) = delete;
        Clique& operator=(Clique&&) = delete;
        Clique(const Clique&) = delete;
        Clique& operator=(const Clique&) = delete;
};

template <int Degree>
class PottsClique : public Clique {
    public:
        PottsClique(const std::vector<NodeId>& nodes, REAL same_cost, REAL diff_cost)
            : m_same_cost(same_cost),
            m_diff_cost(diff_cost)
        { 
            ASSERT(m_diff_cost >= m_same_cost);
            for (int i = 0; i < Degree; ++i)
                m_nodes[i] = nodes[i];
        }

        virtual REAL Energy(const Label* labels) const override {
            const Label l = labels[0];
            for (int i = 1; i < Degree; ++i) {
                if (labels[i] != l)
                    return m_diff_cost;
            }
            return m_same_cost;
        }
        virtual const NodeId* Nodes() const override {
            return m_nodes;
        }
        virtual size_t Size() const override { return Degree; }
        virtual double Rho() const override {
            double f_max = m_diff_cost;
            double f_min;
            if (m_same_cost == 0)
                f_min = m_diff_cost;
            else
                f_min = m_same_cost;
            return f_max / f_min;
        }
    private:
        NodeId m_nodes[Degree];
        REAL m_same_cost;
        REAL m_diff_cost;
};

/*
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
*/
#endif
