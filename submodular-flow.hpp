#ifndef _SUBMODULAR_FLOW_HPP_
#define _SUBMODULAR_FLOW_HPP_

#include <vector>
#include <memory>

typedef int64_t REAL;

class SubmodularFlow {
    public:
        typedef int NodeId;
        struct Clique;
        typedef std::shared_ptr<Clique> CliquePtr;
        typedef std::vector<CliquePtr> CliqueVec;

        SubmodularFlow();

        NodeId AddNode(int n = 1);
        int GetLabel(NodeId n) const;
        void AddUnaryTerm(NodeId n, REAL E0, REAL E1);
        void AddClique(const CliquePtr& cp);

        void PushRelabel();
        void ComputeMinCut();

        /* Clique: abstract base class for user-defined clique functions
         *
         * Clique stores the list of nodes associated with a clique.
         * Actual functionality is provided by the user writing a derived
         * class with Clique as the base, and which implements the
         * ComputeEnergy and ExchangeCapacity functions
         */
        class Clique {
            public:
            Clique(const std::vector<NodeId>& nodes) : m_nodes(nodes) { }
            ~Clique() = default;

            virtual REAL ComputeEnergy(const std::vector<int>& labels) const = 0;
            virtual REAL ExchangeCapacity(NodeId u, NodeId v) const = 0;

            const std::vector<NodeId>& Nodes() const { return m_nodes; }
            size_t Size() const { return m_nodes.size(); }

            protected:
            std::vector<NodeId> m_nodes; // The list of nodes in the clique
            
            // Prohibit copying and moving clique functions, to prevent slicing
            // of derived class data
            Clique(Clique&&) = delete; 
            Clique& operator=(Clique&&) = delete;
            Clique(const Clique&) = delete; 
            Clique& operator=(const Clique&) = delete; 
        };

    protected:
        std::vector<REAL> m_phi_si;
        std::vector<REAL> m_phi_it;
        CliqueVec m_cliques;
};

/*
 * EnergyTableClique: stores energy as a list of 2^k values for each subset
 */
class EnergyTableClique : public SubmodularFlow::Clique {
    public:
        typedef SubmodularFlow::Clique::NodeId NodeId;

        EnergyTableClique(const std::vector<NodeId>& nodes, 
                          const std::vector<REAL>& energy)
            : SubmodularFlow::Clique(nodes),
            m_energy(energy) { }

        virtual REAL ComputeEnergy(const std::vector<int>& labels) const;
        virtual REAL ExchangeCapacity(NodeId u, NodeId v) const;

    protected:
        std::vector<REAL> m_energy;
}

#endif
