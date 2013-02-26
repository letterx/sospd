#include "submodular-flow.hpp"

SubmodularFlow::SubmodularFlow() 
    : m_num_nodes(0),
    m_phi_si(),
    m_phi_it(),
    m_labels(),
    m_num_cliques(0),
    m_cliques(),
    m_neighbors()
{ }

SubmodularFlow::NodeId SubmodularFlow::AddNode(int n) {
    ASSERT(n >= 1);
    NodeId first_node = m_num_nodes;
    for (int i = 0; i < n; ++i) {
        m_phi_si.push_back(0);
        m_phi_it.push_back(0);
        m_labels.push_back(-1);
        m_neighbors.push_back(NeighborList());
        m_num_nodes++;
    }
    return first_node;
}

int SubmodularFlow::GetLabel(NodeId n) const {
    return m_labels[n];
}

void SubmodularFlow::AddUnaryTerm(NodeId n, REAL E0, REAL E1) {
    m_phi_si[n] += E0;
    m_phi_it[n] += E1;
}

void SubmodularFlow::AddClique(const CliquePtr& cp) {
    m_cliques.push_back(cp);
    for (NodeId i : cp->Nodes()) {
        ASSERT(0 <= i && i < m_num_nodes);
        m_neighbors[i].push_back(m_num_cliques);
    }
    m_num_cliques++;
}

void SubmodularFlow::PushRelabel() {
    // Implement me (Sam)
}

void SubmodularFlow::ComputeMinCut() { 
    // Implement me (Sam)
}

REAL EnergyTableClique::ComputeEnergy(const std::vector<int>& labels) const {
    Assignment assgn = 0;
    for (size_t i = 0; i < this->m_nodes.size(); ++i) {
        NodeId n = this->m_nodes[i];
        if (labels[n] == 1) {
            assgn |= 1 << i;
        }
    }
    return m_energy[assgn];
}

REAL EnergyTableClique::ExchangeCapacity(NodeId u, NodeId v) const {
    // This is not the most efficient way to do things, but it works
    const size_t u_idx = std::find(this->m_nodes.begin(), this->m_nodes.end(), u) - this->m_nodes.begin();
    const size_t v_idx = std::find(this->m_nodes.begin(), this->m_nodes.end(), v) - this->m_nodes.end();

    REAL min_energy = std::numeric_limits<REAL>::max();
    Assignment num_assgns = 1 << this->m_nodes.size();
    for (Assignment assgn = 0; assgn < num_assgns; ++assgn) {
        if (assgn & (1 << u_idx) && !(assgn & (1 << v_idx))) {
            // then assgn is a set separating u from v
            if (m_energy[assgn] < min_energy) min_energy = m_energy[assgn];
        }
    }
    return min_energy;
}
