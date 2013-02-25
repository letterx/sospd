#ifndef _SUBMODULAR_FLOW_HPP_
#define _SUBMODULAR_FLOW_HPP_

#include <vector>
#include <memory>

typedef int64_t REAL;

class SubmodularFlowNetwork {
    public:
        typedef int NodeId;
        struct Clique;
        typedef std::shared_ptr<Clique> CliquePtr;
        typedef std::vector<CliquePtr> CliqueVec;

        SubmodularFlowNetwork();

        NodeId AddNode(int n = 1);
        int GetLabel(NodeId n);
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

            REAL ComputeEnergy(const std::vector<int>& labels) const = 0;
            REAL ExchangeCapacity(NodeId u, NodeId v) const = 0;

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

#endif
