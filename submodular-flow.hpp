#ifndef _SUBMODULAR_FLOW_HPP_
#define _SUBMODULAR_FLOW_HPP_

#include "sos-common.hpp"
#include <list>
#include <map>

typedef int64_t REAL;

class SubmodularFlow {
    public:
        typedef int NodeId;
        typedef int CliqueId;
        struct Clique;
        typedef std::shared_ptr<Clique> CliquePtr;
        typedef std::vector<CliquePtr> CliqueVec;
        struct arc {
	        NodeId i, j;
	        CliqueId c; // if this is a clique edge; -1 otherwise
        };
        typedef arc Arc;

        SubmodularFlow();

        // Add n new nodes to the base set V
        NodeId AddNode(int n = 1);

        // GetLabel returns 1, 0 or -1 if n is in S, not in S, or haven't
        // computed flow yet, respectively
        int GetLabel(NodeId n) const;

        // AddUnaryTerm for node n, with cost E0 for not being in S and E1
        // for being in S
        void AddUnaryTerm(NodeId n, REAL E0, REAL E1);

        // AddClique pointed to by cp
        void AddClique(const CliquePtr& cp);

        // Compute the max flow using PushRelabel
        void PushRelabel();
        // After computing the max flow, extract the min cut
        void ComputeMinCut();

        // Compute the total energy across all cliques of the current labeling
        REAL ComputeEnergy() const;

        /* Clique: abstract base class for user-defined clique functions
         *
         * Clique stores the list of nodes associated with a clique.
         * Actual functionality is provided by the user writing a derived
         * class with Clique as the base, and which implements the
         * ComputeEnergy and ExchangeCapacity functions
         */
        class Clique {
            public:
            Clique(const std::vector<NodeId>& nodes)
                : m_nodes(nodes),
                m_alpha_Ci(nodes.size(), 0)
            { }
            ~Clique() = default;

            // Returns the energy of the given labeling for this clique function
            virtual REAL ComputeEnergy(const std::vector<int>& labels) const = 0;
            // Returns the exchange capacity between nodes u and v
            virtual REAL ExchangeCapacity(NodeId u, NodeId v) const = 0;
            // Returns the residual capacity of arc in a particular clique.
            REAL CliqueResidualCapacity(Arc arc);


            const std::vector<NodeId>& Nodes() const { return m_nodes; }
            size_t Size() const { return m_nodes.size(); }
            std::vector<REAL> AlphaCi() { return m_alpha_Ci; }

            protected:
            std::vector<NodeId> m_nodes; // The list of nodes in the clique
            std::vector<REAL> m_alpha_Ci; // The reparameterization variables for this clique

            // Prohibit copying and moving clique functions, to prevent slicing
            // of derived class data
            Clique(Clique&&) = delete;
            Clique& operator=(Clique&&) = delete;
            Clique(const Clique&) = delete;
            Clique& operator=(const Clique&) = delete;
        };

    protected:
        // Layers store vertices by distance.
        struct preflow_layer {
            std::list<NodeId> active_vertices;
            // std::list<NodeId> inactive_vertices;
        };

        typedef preflow_layer Layer;
        typedef std::vector<Layer> LayerArray;
        typedef typename LayerArray::iterator layer_iterator;

        LayerArray layers;
        int max_active, min_active;
        typedef typename std::list<NodeId>::iterator list_iterator;
        std::map<NodeId, typename std::list<NodeId>::iterator> layer_list_ptr;

        NodeId s,t;
        std::map<NodeId,int> dis;
        std::map<NodeId,REAL> excess;
        std::vector<int> current_arc_index;
        std::vector< std::vector<Arc> > m_arc_list;

        void add_to_active_list(NodeId u, Layer& layer);
        void remove_from_active_list(NodeId u);
       Arc FindPushableEdge(NodeId i);
        void Push(Arc arc);
        void Relabel(NodeId i);

    //  protected:
        typedef std::vector<CliqueId> NeighborList;

        NodeId m_num_nodes;
        std::vector<REAL> m_c_si;
        std::vector<REAL> m_c_it;
        std::vector<REAL> m_phi_si;
        std::vector<REAL> m_phi_it;
        std::vector<int> m_labels;

        CliqueId m_num_cliques;
        CliqueVec m_cliques;
        std::vector<NeighborList> m_neighbors;
};

/*
 * EnergyTableClique: stores energy as a list of 2^k values for each subset
 */
class EnergyTableClique : public SubmodularFlow::Clique {
    public:
        typedef SubmodularFlow::NodeId NodeId;
        typedef uint32_t Assignment;

        EnergyTableClique(const std::vector<NodeId>& nodes,
                          const std::vector<REAL>& energy)
            : SubmodularFlow::Clique(nodes),
            m_energy(energy)
        { ASSERT(nodes.size() <= 31); }

        virtual REAL ComputeEnergy(const std::vector<int>& labels) const;
        virtual REAL ExchangeCapacity(NodeId u, NodeId v) const;

    protected:
        std::vector<REAL> m_energy;
};

#endif
