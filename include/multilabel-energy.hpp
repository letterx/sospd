#ifndef _CLIQUE_HPP_
#define _CLIQUE_HPP_

/** \file multilabel-energy.hpp
 * Classes for defining multi-label energy functions, i.e., Markov Random 
 * Fields.
 */

#include "energy-common.hpp"

class Clique;

/** A multilabel energy function, which splits as a sum of clique energies
 *
 * MultilabelEnergy keeps track of a function of the form
 *  \f[ f(x) = \sum_i f_i(x_i) + \sum_C f_C(x_C) \f]
 * where the variables \f$x_i\f$ come from a label set 1,...,L, and there are
 * functions \f$f_C\f$ defined over a set of cliques \f$C\f$ which are subsets
 * of the variables.
 */
class MultilabelEnergy { 
    public:
        typedef int VarId;
        typedef size_t Label;
        typedef std::unique_ptr<Clique> CliquePtr;

        /** Construct an empty energy function with labels 0,...,max_label-1
         */
        MultilabelEnergy(Label max_label);

        /** Add variables to the MRF. Default to add a single variable
         *
         * \param i Number of variables to add
         * \return VarId of first variable added
         */
        VarId addVar(int i = 1);

        /** Add a constant term (independent of labeling) to the function
         */
        void addConstantTerm(REAL c);

        /** Add a unary term (depending on a single variable) to the function
         *
         * \param i Variable to add unary term for
         * \param coeffs Vector of costs for each labeling of x_i. Must be of
         * length max_label
         */
        void addUnaryTerm(VarId i, const std::vector<REAL>& coeffs);

        /** Add a clique function to the energy
         *
         * Because Clique is polymorphic, these must be constructed on the 
         * heap, MultilabelEnergy takes ownership (via unique_ptr).
         */
        void addClique(CliquePtr c);

        VarId numVars() const { return m_numVars; }
        size_t numCliques() const { return m_cliques.size(); }
        Label numLabels() const { return m_maxLabel; }
        REAL computeEnergy(const std::vector<Label>& labels) const;
        const std::vector<CliquePtr>& cliques() const { return m_cliques; }
        REAL unary(VarId i, Label l) const { return m_unary[i][l]; }
        REAL& unary(VarId i, Label l) { return m_unary[i][l]; }

    protected:
        const Label m_maxLabel;
        VarId m_numVars;
        REAL m_constantTerm;
        std::vector<std::vector<REAL>> m_unary;
        std::vector<CliquePtr> m_cliques;
        
    private:
        MultilabelEnergy() = delete;
};

class Clique {
    public:   
        typedef MultilabelEnergy::VarId VarId;
        typedef MultilabelEnergy::Label Label;
        Clique() { }
        virtual ~Clique() = default;

        // labels must be an array of length this->size() with the labels
        // corresponding to the nodes.
        // Returns the energy of the clique at that labeling
        virtual REAL energy(const Label* labels) const = 0;
        virtual const VarId* nodes() const = 0;
        virtual size_t size() const = 0;

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
        PottsClique(const std::vector<VarId>& nodes, REAL same_cost, 
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
        virtual const VarId* nodes() const override {
            return m_nodes;
        }
        virtual size_t size() const override { return Degree; }
    private:
        VarId m_nodes[Degree];
        REAL m_sameCost;
        REAL m_diffCost;
};


/********* Multilabel Implementation ***************/

inline MultilabelEnergy::MultilabelEnergy(Label max_label)
    : m_maxLabel(max_label),
    m_numVars(0),
    m_constantTerm(0),
    m_unary(),
    m_cliques()
{ }

inline MultilabelEnergy::VarId MultilabelEnergy::addVar(int i) {
    VarId ret = m_numVars;
    for (int j = 0; j < i; ++j) {
        m_unary.push_back(std::vector<REAL>(m_maxLabel, 0));
    }
    m_numVars += i;
    return ret;
}

inline void MultilabelEnergy::addConstantTerm(REAL c) {
    m_constantTerm++;
}

inline void MultilabelEnergy::addUnaryTerm(VarId i,
        const std::vector<REAL>& coeffs) {
    ASSERT(i < m_numVars);
    ASSERT(Label(coeffs.size()) == m_maxLabel);
    for (Label l = 0; l < coeffs.size(); ++l)
        m_unary[i][l] += coeffs[l];
}

inline void MultilabelEnergy::addClique(CliquePtr c) {
    m_cliques.push_back(std::move(c));
}

inline REAL 
MultilabelEnergy::computeEnergy(const std::vector<Label>& labels) const {
    ASSERT(VarId(labels.size()) == m_numVars);
    REAL energy = 0;
    std::vector<Label> label_buf;
    for (const CliquePtr& cp : m_cliques) {
        int k = cp->size();
        label_buf.resize(k);
        const VarId* nodes = cp->nodes();
        for (int i = 0; i < k; ++i)
            label_buf[i] = labels[nodes[i]];
        energy += cp->energy(label_buf.data());
    }
    for (VarId i = 0; i < m_numVars; ++i)
        energy += m_unary[i][labels[i]];
    return energy;
}

#endif
