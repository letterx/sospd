#ifndef _SUBMODULAR_FLOW_HPP_
#define _SUBMODULAR_FLOW_HPP_

#include "energy-common.hpp"
#include "boost/optional/optional.hpp"
#include <list>

class EnergyTableClique;

class SubmodularFlow {
    public:
        typedef size_t NodeId;
        //typedef size_t CliqueId;
        typedef int CliqueId;
        struct Clique;
        typedef std::shared_ptr<EnergyTableClique> CliquePtr;
        typedef std::vector<CliquePtr> CliqueVec;
        struct arc {
	        NodeId i, j;
            size_t i_idx, j_idx;
	        CliqueId c; // if this is a clique edge; -1 otherwise
            REAL cached_cap; // Is the true capacity if cached_time == c.m_time
            int64_t cache_time;
        };
        typedef arc Arc;

        SubmodularFlow();

        // Add n new nodes to the base set V
        NodeId AddNode(int n = 1);

        // GetLabel returns 1, 0 or -1 if n is in S, not in S, or haven't
        // computed flow yet, respectively
        int GetLabel(NodeId n) const;

        // Add a constant to the energy function
        void AddConstantTerm(REAL c) { m_constant_term += c; }

        // AddUnaryTerm for node n, with cost E0 for not being in S and E1
        // for being in S
        void AddUnaryTerm(NodeId n, REAL E0, REAL E1);
        void AddUnaryTerm(NodeId n, REAL coeff);

        // Add Clique pointed to by cp
        void AddClique(const CliquePtr& cp);
        // Add Clique defined by nodes and energy table given
        void AddClique(const std::vector<NodeId>& nodes, const std::vector<REAL>& energyTable);
        void AddPairwiseTerm(NodeId i, NodeId j, REAL E00, REAL E01, REAL E10, REAL E11);

        // Compute the max flow using PushRelabel
        void PushRelabel();
        // After computing the max flow, extract the min cut
        void ComputeMinCut();
        // Do both the above
        void Solve();

        // Compute the total energy across all cliques of the current labeling
        REAL ComputeEnergy() const;
        REAL ComputeEnergy(const std::vector<int>& labels) const;

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
                m_alpha_Ci(nodes.size(), 0),
                m_time(0)
            { }
            ~Clique() = default;

            // Returns the energy of the given labeling for this clique function
            virtual REAL ComputeEnergy(const std::vector<int>& labels) const = 0;
            /*
            // Returns the exchange capacity between nodes u and v
            virtual REAL ExchangeCapacity(size_t u_idx, size_t v_idx) const = 0;
            virtual bool NonzeroCapacity(size_t u_idx, size_t v_idx) const = 0;
            // Normalizes energy so that it is always >= 0, and the all 1 and
            // all 0 labeling have energy 0. Subtracts a linear function from
            // the energy, so we may need to change c_si, c_it
            virtual void NormalizeEnergy(SubmodularFlow& sf) = 0;
            // Push delta units of flow from u to v
            virtual void Push(size_t u_idx, size_t v_idx, REAL delta) = 0;
            */

            const std::vector<NodeId>& Nodes() const { return m_nodes; }
            size_t Size() const { return m_nodes.size(); }
            std::vector<REAL>& AlphaCi() { return m_alpha_Ci; }
            const std::vector<REAL>& AlphaCi() const { return m_alpha_Ci; }
            size_t GetIndex(NodeId i) const {
                return std::find(this->m_nodes.begin(), this->m_nodes.end(), i) - this->m_nodes.begin();
            }
            REAL ComputeEnergyAlpha(const std::vector<int>& labels) const {
                REAL e = ComputeEnergy(labels);
                for (size_t idx = 0; idx < m_nodes.size(); ++idx) {
                    if (labels[m_nodes[idx]] == 1) e -= m_alpha_Ci[idx];
                }
                return e;
            }
            int64_t& Time() { return m_time; }
            int64_t Time() const { return m_time; }

            protected:
            std::vector<NodeId> m_nodes; // The list of nodes in the clique
            std::vector<REAL> m_alpha_Ci; // The reparameterization variables for this clique
            int64_t m_time;

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
            //std::list<NodeId> inactive_vertices;
        };

        typedef preflow_layer Layer;
        typedef std::vector<Layer> LayerArray;
        typedef typename LayerArray::iterator layer_iterator;

        LayerArray layers;
        int max_active, min_active;
        typedef typename std::list<NodeId>::iterator list_iterator;
        std::vector<list_iterator> layer_list_ptr;

        // Data needed during push-relabel
        NodeId s,t;
        std::vector<int> dis;
        std::vector<REAL> excess;
        typedef std::vector<Arc> ArcList;
        std::vector<typename ArcList::iterator> current_arc;
        std::vector< ArcList > m_arc_list;
        long work_since_last_update;
        long num_edges;

        void add_to_active_list(NodeId u, Layer& layer);
        void remove_from_active_list(NodeId u);
        REAL ResCap(Arc& arc);
        bool NonzeroCap(Arc& arc);
        Arc* FindPushableEdge(NodeId i);
        void Push(Arc& arc);
        void Relabel(NodeId i);
        void Discharge(NodeId i);

        typedef std::vector<CliqueId> NeighborList;

        REAL m_constant_term;
        NodeId m_num_nodes;
        std::vector<REAL> m_c_si;
        std::vector<REAL> m_c_it;
        std::vector<REAL> m_phi_si;
        std::vector<REAL> m_phi_it;
        std::vector<int> m_labels;

        CliqueId m_num_cliques;
        CliqueVec m_cliques;
        std::vector<NeighborList> m_neighbors;

        size_t m_num_clique_pushes;
        size_t m_num_global_relabels;

    public:
        // Functions for reading out data, useful for testing
        NodeId GetS() const { return s; }
        NodeId GetT() const { return t; }
        const std::vector<int>& GetDis() const { return dis; }
        const std::vector<REAL>& GetExcess() const { return excess; }
        REAL GetConstantTerm() const { return m_constant_term; }
        NodeId GetNumNodes() const { return m_num_nodes; }
        const std::vector<REAL>& GetC_si() const { return m_c_si; }
        const std::vector<REAL>& GetC_it() const { return m_c_it; }
        const std::vector<REAL>& GetPhi_si() const { return m_phi_si; }
        const std::vector<REAL>& GetPhi_it() const { return m_phi_it; }
        const std::vector<int>& GetLabels() const { return m_labels; }
        std::vector<int>& GetLabels() { return m_labels; }
        CliqueId GetNumCliques() const { return m_num_cliques; }
        const CliqueVec& GetCliques() const { return m_cliques; }
        const std::vector<NeighborList>& GetNeighbors() const { return m_neighbors; }
        int GetMaxActive() const { return max_active; }
        int GetMinActive() const { return min_active; }

        // Functions for breaking push-relabel into pieces
        // Only use for testing!
        void PushRelabelInit();
        void PushRelabelStep();
        bool PushRelabelNotDone();
        void GlobalRelabel();
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
            m_energy(energy),
            m_alpha_energy(energy),
            m_min_tight_set(nodes.size(), (1 << nodes.size()) - 1)
        { 
            ASSERT(nodes.size() <= 31); 
        }

        // Virtual overrides
        virtual REAL ComputeEnergy(const std::vector<int>& labels) const;
        REAL ExchangeCapacity(size_t u_idx, size_t v_idx) const;
        bool NonzeroCapacity(size_t u_idx, size_t v_idx) const;
        void NormalizeEnergy(SubmodularFlow& sf);
        void Push(size_t u_idx, size_t v_idx, REAL delta);

        void ComputeMinTightSets();
        void EnforceSubmodularity();

    protected:
        std::vector<REAL> m_energy;
        std::vector<REAL> m_alpha_energy;
        std::vector<Assignment> m_min_tight_set;
};


#endif
