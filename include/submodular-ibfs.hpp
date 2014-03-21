#ifndef _SUBMODULAR_IBFS_HPP_
#define _SUBMODULAR_IBFS_HPP_

#include "energy-common.hpp"
#include <list>
#include <algorithm>
#include <boost/shared_ptr.hpp>
#include <boost/intrusive/list.hpp>

class IBFSEnergyTableClique;

class SubmodularIBFS {
    public:
        typedef int NodeId;
        typedef int CliqueId;
        typedef std::vector<CliqueId> NeighborList;
        enum class NodeState : char {
            S, T, S_orphan, T_orphan, N
        };
        typedef std::list<NodeId> NodeIdList;


        SubmodularIBFS();

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
        void ClearUnaries();

        // Add Clique defined by nodes and energy table given
        void AddClique(const std::vector<NodeId>& nodes, const std::vector<REAL>& energyTable) {
            AddClique(nodes, energyTable, true);
        }
        void AddClique(const std::vector<NodeId>& nodes, const std::vector<REAL>& energyTable, bool normalize);
        void AddPairwiseTerm(NodeId i, NodeId j, REAL E00, REAL E01, REAL E10, REAL E11);

        void IBFS();
        void GraphInit();
        void ComputeMinCut();
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
            typedef std::vector<NodeId> NodeVec;
            Clique() : m_nodes(), m_alpha_Ci(), m_time(0) { }
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
            virtual void NormalizeEnergy(SubmodularIBFS& sf) = 0;
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
            NodeVec m_nodes; // The list of nodes in the clique
            std::vector<REAL> m_alpha_Ci; // The reparameterization variables for this clique
            int64_t m_time;

        };
        /*
         * IBFSEnergyTableClique: stores energy as a list of 2^k values for each subset
         */
        class IBFSEnergyTableClique : public Clique {
            public:
                typedef SubmodularIBFS::NodeId NodeId;
                typedef uint32_t Assignment;

                IBFSEnergyTableClique() : SubmodularIBFS::Clique(), m_energy(), m_alpha_energy(), m_min_tight_set() { }
                IBFSEnergyTableClique(const std::vector<NodeId>& nodes,
                                  const std::vector<REAL>& energy)
                    : SubmodularIBFS::Clique(nodes),
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
                void NormalizeEnergy(SubmodularIBFS& sf);
                void Push(size_t u_idx, size_t v_idx, REAL delta);

                void ComputeMinTightSets();
                void EnforceSubmodularity();
                std::vector<REAL>& EnergyTable() { return m_energy; }
                const std::vector<REAL>& EnergyTable() const { return m_energy; }

                void ResetAlpha();

            protected:
                std::vector<REAL> m_energy;
                std::vector<REAL> m_alpha_energy;
                std::vector<Assignment> m_min_tight_set;

        };
        struct ArcIterator {
            NodeId source;
            NeighborList::iterator cIter;
            int cliqueIdx;
            int cliqueSize;
            SubmodularIBFS* ibfs;
            
            bool operator!=(const ArcIterator& a) {
                return (cIter != a.cIter) || (cliqueIdx != a.cliqueIdx);
            }
            bool operator==(const ArcIterator& a) {
                return !(*this != a);
            }
            bool operator<(const ArcIterator& a) {
                ASSERT(source == a.source);
                return (cIter == a.cIter) ? (source < a.source) : (cIter < a.cIter);
            }

            ArcIterator& operator++() {
                //ASSERT(*cIter < static_cast<int>(ibfs->m_cliques.size()));
                cliqueIdx++;
                if (cliqueIdx == cliqueSize) {
                    cliqueIdx = 0;
                    cIter++;
                    if (cIter != ibfs->m_neighbors[source].end())
                        cliqueSize = ibfs->m_cliques[*cIter].Size();
                    else
                        cliqueSize = 0;
                }
                //ASSERT(cIter == ibfs->m_neighbors[source].end() || *cIter < static_cast<int>(ibfs->m_cliques.size()));
                //ASSERT(cIter == ibfs->m_neighbors[source].end() || cliqueIdx < static_cast<int>(ibfs->m_cliques[*cIter].Nodes().size()));
                return *this;
            }
            NodeId Source() const {
                return source;
            }
            NodeId Target() const {
                ASSERT(*cIter < static_cast<int>(ibfs->m_cliques.size()));
                ASSERT(cliqueIdx < static_cast<int>(ibfs->m_cliques[*cIter].Nodes().size()));
                return ibfs->m_cliques[*cIter].Nodes()[cliqueIdx];
            }
            int SourceIdx() const { return ibfs->m_cliques[*cIter].GetIndex(source); }
            int TargetIdx() const { return cliqueIdx; }
            CliqueId CliqueId() const { return *cIter; }
            ArcIterator Reverse() const {
                auto newSource = Target();
                auto newCIter = std::find(ibfs->m_neighbors[newSource].begin(), ibfs->m_neighbors[newSource].end(), *cIter);
                auto newCliqueIdx = ibfs->GetCliques()[*newCIter].GetIndex(source);
                return {newSource, newCIter, static_cast<int>(newCliqueIdx), static_cast<int>(ibfs->m_cliques[*newCIter].Size()), ibfs};
            }
        };

        typedef boost::intrusive::list_base_hook<boost::intrusive::link_mode<boost::intrusive::normal_link>> ListHook;
        struct Node : public ListHook {
            NodeState state;
            int dis;
            ArcIterator parent_arc;
            NodeId parent;
            NeighborList cliques;
            Node() : state(NodeState::N), dis(std::numeric_limits<int>::max()), parent_arc(), cliques() { }
        };

        typedef boost::intrusive::list<Node> NodeQueue;

        ArcIterator ArcsBegin(NodeId i) {
            auto cIter = m_neighbors[i].begin();
            if (cIter == m_neighbors[i].end())
                return ArcsEnd(i);
            return {i, cIter, 0, static_cast<int>(m_cliques[*cIter].Size()), this};
        }
        ArcIterator ArcsEnd(NodeId i) {
            auto& neighborList = m_neighbors[i];
            return {i, neighborList.end(), 0, 0, this};
        }

        typedef std::vector<IBFSEnergyTableClique> CliqueVec;

    protected:
        // Layers store vertices by distance.
        std::vector<NodeQueue> m_source_layers;
        std::vector<NodeQueue> m_sink_layers;
        NodeIdList m_source_orphans;
        NodeIdList m_sink_orphans;
        int m_source_tree_d;
        int m_sink_tree_d;
        typedef typename NodeQueue::iterator queue_iterator;
        queue_iterator m_search_node_iter;
        queue_iterator m_search_node_end;
        ArcIterator m_search_arc;
        ArcIterator m_search_arc_end;
        bool m_forward_search;

        // Data needed during push-relabel
        NodeId s,t;
        long num_edges;

        void Push(ArcIterator& arc, bool forwardArc, REAL delta);
        void Augment(ArcIterator& arc);
        void Adopt();
        void MakeOrphan(NodeId i);
        void RemoveFromLayer(NodeId i);
        void AddToLayer(NodeId i);
        void AdvanceSearchNode();

        void IBFSInit();


        REAL m_constant_term;
        NodeId m_num_nodes;
        std::vector<Node> m_nodes;
        std::vector<REAL> m_c_si;
        std::vector<REAL> m_c_it;
        std::vector<REAL> m_phi_si;
        std::vector<REAL> m_phi_it;
        std::vector<int> m_labels;

        CliqueId m_num_cliques;
        CliqueVec m_cliques;
        std::vector<NeighborList> m_neighbors;

        size_t m_num_clique_pushes;

        double m_totalTime = 0;
        double m_graphInitTime = 0;
        double m_initTime = 0;
        double m_augmentTime = 0;
        double m_adoptTime = 0;

    public:
        // Functions for reading out data, useful for testing
        NodeId GetS() const { return s; }
        NodeId GetT() const { return t; }
        REAL GetConstantTerm() const { return m_constant_term; }
        NodeId GetNumNodes() const { return m_num_nodes; }
        const std::vector<REAL>& GetC_si() const { return m_c_si; }
        const std::vector<REAL>& GetC_it() const { return m_c_it; }
        const std::vector<REAL>& GetPhi_si() const { return m_phi_si; }
        const std::vector<REAL>& GetPhi_it() const { return m_phi_it; }
        CliqueId GetNumCliques() const { return m_num_cliques; }
        const CliqueVec& GetCliques() const { return m_cliques; }
        CliqueVec& GetCliques() { return m_cliques; }
        const std::vector<NeighborList>& GetNeighbors() const { return m_neighbors; }
        std::vector<int>& GetLabels() { return m_labels; }
        const std::vector<int>& GetLabels() const { return m_labels; }
        const std::vector<Node>& GetNodes() const { return m_nodes; }

        REAL ResCap(const ArcIterator& arc, bool forwardArc);
        bool NonzeroCap(const ArcIterator& arc, bool forwardArc);

};


#endif
