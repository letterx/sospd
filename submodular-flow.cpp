#include <cstdio>
#include "submodular-flow.hpp"
#include <algorithm>
#include <limits>
#include <queue>

SubmodularFlow::SubmodularFlow()
    : m_constant_term(0),
    m_num_nodes(0),
    m_c_si(),
    m_c_it(),
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
        m_c_si.push_back(0);
        m_c_it.push_back(0);
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
    // Reparametize so that E0, E1 >= 0
    if (E0 < 0) {
        AddConstantTerm(E0);
        E0 = 0;
        E1 -= E0;
    }
    if (E1 < 0) {
        AddConstantTerm(E1);
        E1 = 0;
        E0 -= E1;
    }
    m_c_si[n] += E0;
    m_c_it[n] += E1;
}

void SubmodularFlow::AddUnaryTerm(NodeId n, REAL coeff) {
    AddUnaryTerm(n, 0, coeff);
}

void SubmodularFlow::AddClique(const CliquePtr& cp) {
    m_cliques.push_back(cp);
    for (NodeId i : cp->Nodes()) {
        ASSERT(0 <= i && i < m_num_nodes);
        m_neighbors[i].push_back(m_num_cliques);
    }
    m_num_cliques++;
    cp->NormalizeEnergy(*this);
}

void SubmodularFlow::AddClique(const std::vector<NodeId>& nodes, const std::vector<REAL>& energyTable) {
    CliquePtr cp(new EnergyTableClique(nodes, energyTable));
    AddClique(cp);
}

//////// Push Relabel methods ///////////

void SubmodularFlow::add_to_active_list(NodeId u, Layer& layer) {
    // BOOST_USING_STD_MIN();
    // BOOST_USING_STD_MAX();
    layer.active_vertices.push_front(u);
    max_active = std::max BOOST_PREVENT_MACRO_SUBSTITUTION(dis[u], max_active);
    min_active = std::min BOOST_PREVENT_MACRO_SUBSTITUTION(dis[u], min_active);
    layer_list_ptr[u] = layer.active_vertices.begin();
}

void SubmodularFlow::remove_from_active_list(NodeId u) {
    layers[dis[u]].active_vertices.erase(layer_list_ptr[u]);
}

void SubmodularFlow::PushRelabelInit()
{
    // super source and sink
    s = m_num_nodes; t = m_num_nodes + 1;
    max_active = 0; min_active = m_num_nodes + 2; // n

    dis.clear(); excess.clear(); current_arc_index.clear();
    m_arc_list.clear(); layers.clear();

    // init data structures
    for (int i = 0; i < m_num_nodes + 2; ++i) {
        dis.push_back(0);
        excess.push_back(0);
        current_arc_index.push_back(0);
        std::vector<Arc> arc_list;
        m_arc_list.push_back(arc_list);
        Layer layer;
        layers.push_back(layer);
    }
    dis[s] = m_num_nodes + 2; // n = m_num_nodes + 2

    // saturate arcs out of s.
    for (NodeId i = 0; i < m_num_nodes; ++i) {
        m_phi_si[i] = m_c_si[i];
        if (m_c_si[i] > 0) {
            excess[s] -= m_c_si[i];
            excess[i] += m_c_si[i];
	        add_to_active_list(i, layers[0]);
        }
    }

    // initialize arc lists
    Arc arc;
    for (NodeId i = 0; i < m_num_nodes; ++i) {
        // arcs from source
        arc.i = s;
        arc.j = i;
        arc.c = -1;
        m_arc_list[arc.i].push_back(arc);

        // arcs to source
        arc.i = i;
        arc.j = s;
        m_arc_list[arc.i].push_back(arc);

        // arcs to sink
        arc.j = t;
        m_arc_list[arc.i].push_back(arc);

        // arcs from sink
        arc.i = t;
        arc.j = i;
        m_arc_list[arc.i].push_back(arc);
    }

    // Arcs between nodes of clique
    for (int cid = 0; cid < m_num_cliques; ++cid) {
        CliquePtr cp = m_cliques[cid];
        for (NodeId i : cp->Nodes()) {
            for (NodeId j : cp->Nodes()) {
                arc.i = i;
                arc.j = j;
                arc.c = cid;
                m_arc_list[i].push_back(arc);
            }
        }
    }
}

void SubmodularFlow::PushRelabelStep()
{
    Layer& layer = layers[max_active];
    list_iterator u_iter = layer.active_vertices.begin();

    if (u_iter == layer.active_vertices.end())
        --max_active;
    else {
        NodeId i = *u_iter;
        remove_from_active_list(i);
        boost::optional<Arc> arc = FindPushableEdge(i);
        if (arc)
            Push(*arc);
        else
            Relabel(i);
    }
}

bool SubmodularFlow::PushRelabelNotDone()
{
    return max_active >= min_active;
}

void SubmodularFlow::PushRelabel()
{
    PushRelabelInit();

    // find active i w/ largest distance
    while (PushRelabelNotDone()) {
        PushRelabelStep();
    }
}

REAL SubmodularFlow::ResCap(Arc arc) {
    if (arc.j == s) {
        return m_phi_si[arc.i];
    } else if (arc.j == t) {
        return m_c_it[arc.i] - m_phi_it[arc.i];
    } else {
        return m_cliques[arc.c]->ExchangeCapacity(arc.i, arc.j);
    }
}

boost::optional<SubmodularFlow::Arc> SubmodularFlow::FindPushableEdge(NodeId i) {
    // Use current arc?
    for (Arc arc : m_arc_list[i]) {
        if (dis[i] == dis[arc.j] + 1 && ResCap(arc) > 0) {
	        return boost::optional<SubmodularFlow::Arc>(arc);
        }
    }
    return boost::optional<SubmodularFlow::Arc>();
}

void SubmodularFlow::Push(Arc arc) {
    REAL delta; // amount to push

    // Note, we never have arc.i == s or t
    if (arc.j == s) {  // reverse arc
        delta = std::min(excess[arc.i], m_phi_si[arc.i]);
        m_phi_si[arc.i] -= delta;
    } else if (arc.j == t) {
        delta = std::min(excess[arc.i], m_c_it[arc.i] - m_phi_it[arc.i]);
        m_phi_it[arc.i] += delta;
    } else { // Clique arc
        delta = std::min(excess[arc.i], m_cliques[arc.c]->ExchangeCapacity(arc.i, arc.j));
        std::vector<REAL>& alpha_ci = m_cliques[arc.c]->AlphaCi();
        alpha_ci[arc.i] += delta;
        alpha_ci[arc.j] -= delta;
    }
    // Update (residual capacities) and excesses
    excess[arc.i] -= delta;
    excess[arc.j] += delta;
    if (excess[arc.j] > 0) {
        remove_from_active_list(arc.j);
        add_to_active_list(arc.j, layers[dis[arc.j]]);
    }
    if (excess[arc.i] > 0) {
        // remove_from_active_list(arc.i);
        add_to_active_list(arc.i, layers[dis[arc.i]]);
    }
}

void SubmodularFlow::Relabel(NodeId i) {
    dis[i] = std::numeric_limits<int>::max();
    for(Arc arc : m_arc_list[i]) {
        if (ResCap(arc) > 0) {
            dis[i] = std::min (dis[i], dis[arc.j] + 1);
        }
    }
    // remove_from_active_list(i);
    add_to_active_list(i, layers[dis[i]]);
}

///////////////    end of push relabel    ///////////////////

void SubmodularFlow::ComputeMinCut() {
    for (NodeId i = 0; i < m_num_nodes; ++i) {
        dis[i] = std::numeric_limits<int>::max();
        m_labels[i] = 1;
    }
    dis[t] = 0;
    // curr is the current level of nodes to be visited;
    // next is the next layer to visit.
    std::queue<NodeId> curr, next;
    next.push(t);

    int level = 1;
    while (!next.empty()) {
        // Next becomes curr; empty next
        std::swap(curr, next);
        std::queue<NodeId> empty;
        std::swap(next, empty);

        while (!curr.empty()) {
            NodeId u = curr.front();
            curr.pop();
            for (Arc arc : m_arc_list[u]) {
                arc.i = arc.j;
                arc.j = u;
                if (ResCap(arc) > 0
                        && dis[arc.i] == std::numeric_limits<int>::max()) {
                    m_labels[arc.i] = 0;
                    next.push(arc.i);
                    dis[arc.i] = level;
                }
            }
        }
        ++level;
    }
}

REAL SubmodularFlow::ComputeEnergy() const {
    REAL total = m_constant_term;
    for (NodeId i = 0; i < m_num_nodes; ++i) {
        if (m_labels[i] == 1) total += m_c_it[i];
        else total += m_c_si[i];
    }
    for (const CliquePtr& cp : m_cliques) {
        total += cp->ComputeEnergy(m_labels);
    }
    return total;
}

void EnergyTableClique::NormalizeEnergy(SubmodularFlow& sf) {
    const size_t n = this->m_nodes.size();
    const Assignment num_assignments = 1 << n;
    const REAL constant_term = m_energy[num_assignments - 1];
    std::vector<REAL> marginals;
    Assignment assgn = num_assignments - 1; // The all 1 assignment
    for (size_t i = 0; i < n; ++i) {
        Assignment next_assgn = assgn ^ (1 << i);
        marginals.push_back(m_energy[assgn] - m_energy[next_assgn]);
        assgn = next_assgn;
    }

    for (Assignment a = 0; a < num_assignments; ++a) {
        m_energy[a] -= constant_term;
        for (size_t i = 0; i < n; ++i) {
            if (!(a & (1 << i))) m_energy[a] += marginals[i];
        }
    }

    sf.AddConstantTerm(constant_term);
    for (size_t i = 0; i < n; ++i) {
        sf.AddUnaryTerm(this->m_nodes[i], -marginals[i], 0);
    }
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
        REAL alpha_C = 0;
        for (size_t i = 0; i < this->m_alpha_Ci.size(); ++i) {
            if (assgn & (1 << i)) alpha_C += this->m_alpha_Ci[i];
        }
        if (assgn & (1 << u_idx) && !(assgn & (1 << v_idx))) {
            // then assgn is a set separating u from v
            REAL energy = m_energy[assgn] - alpha_C;
            if (energy < min_energy) min_energy = energy;
        }
    }
    return min_energy;
}
