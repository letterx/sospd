#ifndef _CLIQUE_HPP_
#define _CLIQUE_HPP_

#include "energy-common.hpp"

typedef int NodeId;
typedef size_t Label;

class Clique {
    public:   
        Clique() { }
        virtual ~Clique() = default;

        // labels must be an array of length this->size() with the labels
        // corresponding to the nodes.
        // Returns the energy of the clique at that labeling
        virtual REAL energy(const Label* labels) const = 0;
        virtual const NodeId* nodes() const = 0;
        virtual size_t size() const = 0;
        virtual double rho() const {
            return std::numeric_limits<double>::infinity(); 
        }
        virtual REAL fMax() const { 
            return std::numeric_limits<REAL>::max(); 
        }

    private:
        // Remove move and copy operators to prevent slicing of base classes
        Clique(Clique&&) = delete;
        Clique& operator=(Clique&&) = delete;
        Clique(const Clique&) = delete;
        Clique& operator=(const Clique&) = delete;
};

class MultilabelEnergy { 
    public:
        typedef ::NodeId NodeId;
        typedef ::Label Label;
        typedef std::unique_ptr<Clique> CliquePtr;

        MultilabelEnergy() = delete;
        MultilabelEnergy(Label max_label);

        NodeId addNode(int i = 1);
        void addConstantTerm(REAL c);
        void addUnaryTerm(NodeId i, const std::vector<REAL>& coeffs);
        void addClique(Clique* c);

        NodeId numNodes() const { return m_numNodes; }
        size_t numCliques() const { return m_cliques.size(); }
        Label numLabels() const { return m_maxLabel; }
        double rho() const;
        REAL computeEnergy(const std::vector<Label>& labels) const;
        const std::vector<CliquePtr>& cliques() const { return m_cliques; }
        REAL unary(NodeId i, Label l) const { return m_unary[i][l]; }
        REAL& unary(NodeId i, Label l) { return m_unary[i][l]; }

    protected:
        Label m_maxLabel;
        NodeId m_numNodes;
        REAL m_constantTerm;
        std::vector<std::vector<REAL>> m_unary;
        std::vector<CliquePtr> m_cliques;
};

template <int Degree>
class PottsClique : public Clique {
    public:
        PottsClique(const std::vector<NodeId>& nodes, REAL same_cost, 
                REAL diff_cost)
            : m_sameCost(same_cost),
            m_diffCost(diff_cost)
        { 
            ASSERT(m_diffCost >= m_sameCost);
            for (int i = 0; i < Degree; ++i)
                m_nodes[i] = nodes[i];
        }

        virtual REAL energy(const Label* labels) const override {
            const Label l = labels[0];
            for (int i = 1; i < Degree; ++i) {
                if (labels[i] != l)
                    return m_diffCost;
            }
            return m_sameCost;
        }
        virtual const NodeId* nodes() const override {
            return m_nodes;
        }
        virtual size_t size() const override { return Degree; }
        virtual double rho() const override {
            double f_max = m_diffCost;
            double f_min;
            if (m_sameCost == 0)
                f_min = m_diffCost;
            else
                f_min = m_sameCost;
            return f_max / f_min;
        }
    private:
        NodeId m_nodes[Degree];
        REAL m_sameCost;
        REAL m_diffCost;
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

/********* Multilabel Implementation ***************/

inline MultilabelEnergy::MultilabelEnergy(Label max_label)
    : m_maxLabel(max_label),
    m_numNodes(0),
    m_constantTerm(0),
    m_unary(),
    m_cliques()
{ }

inline NodeId MultilabelEnergy::addNode(int i) {
    NodeId ret = m_numNodes;
    for (int j = 0; j < i; ++j) {
        m_unary.push_back(std::vector<REAL>(m_maxLabel, 0));
    }
    m_numNodes += i;
    return ret;
}

inline void MultilabelEnergy::addConstantTerm(REAL c) {
    m_constantTerm++;
}

inline void MultilabelEnergy::addUnaryTerm(NodeId i,
        const std::vector<REAL>& coeffs) {
    ASSERT(i < m_numNodes);
    ASSERT(Label(coeffs.size()) == m_maxLabel);
    for (Label l = 0; l < coeffs.size(); ++l)
        m_unary[i][l] += coeffs[l];
}

inline void MultilabelEnergy::addClique(Clique* c) {
    m_cliques.push_back(CliquePtr(c));
}

inline double MultilabelEnergy::rho() const {
    double rho = 0;
    for (const CliquePtr& cp : m_cliques)
        rho = std::max(rho, cp->rho());
    return rho;
}

inline REAL 
MultilabelEnergy::computeEnergy(const std::vector<Label>& labels) const {
    ASSERT(NodeId(labels.size()) == m_numNodes);
    REAL energy = 0;
    std::vector<Label> label_buf;
    for (const CliquePtr& cp : m_cliques) {
        int k = cp->size();
        label_buf.resize(k);
        const NodeId* nodes = cp->nodes();
        for (int i = 0; i < k; ++i)
            label_buf[i] = labels[nodes[i]];
        energy += cp->energy(label_buf.data());
    }
    for (NodeId i = 0; i < m_numNodes; ++i)
        energy += m_unary[i][labels[i]];
    return energy;
}

#endif
