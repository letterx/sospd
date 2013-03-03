#include "submodular-flow.hpp"
#include <boost/heap/binomial_heap.hpp>
#include <algorithm>

SubmodularFlow::SubmodularFlow()
    : m_num_nodes(0),
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
    m_c_si[n] += E0;
    m_c_it[n] += E1;
}

void SubmodularFlow::AddClique(const CliquePtr& cp) {
    m_cliques.push_back(cp);
    for (NodeId i : cp->Nodes()) {
        ASSERT(0 <= i && i < m_num_nodes);
        m_neighbors[i].push_back(m_num_cliques);
    }
    m_num_cliques++;
}

/* Deprecated..
// sums up alpha values of cliques that node i belongs to
REAL SumAlphaCi(NodeId i) {
    REAL total = 0;
    for(CliqueVec::iterator it = m_cliques[i].begin(); it != m_cliques[i].end(); ++it) {
        CliquePtr clique = *it;
        for (std::vector<NodeId>::iterator nodei = clique->m_nodes.begin();
                nodei != clique->m_nodes.end(); ++nodei) {
            if (*nodei == i) {
                total += clique->m_alpha_ci[*nodei];
                break;
            }
        }
    }
    return total;
}*/

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

void SubmodularFlow::PushRelabel()
{
    // Super source and sink
    s = m_num_nodes;
    t = m_num_nodes + 1;

    // init dis
    for (int i = 0; i <= m_num_nodes + 1; ++i) {
	dis[i] = 0;
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

    // find active i w/ largest distance
    while (max_active >= min_active) {
        Layer& layer = layers[max_active];
        list_iterator u_iter = layer.active_vertices.begin();

        if (u_iter == layer.active_vertices.end())
            --max_active;
        else {
            NodeId i = *u_iter;
            remove_from_active_list(i);
	    Arc arc = FindPushableEdge(i);
	    if (arc == NULL)
	        Push(arc);
	    else
	        Relabel(i);
        }
    }
}

Arc SubmodularFlow::FindPushableEdge(NodeId i) {
    for(std::vector<Arc>::iterator it = m_arc_list[i].begin();
            it != m_arc_list[i].end(); ++it) {
        Arc arc = *it;
        if (d[i] = d[arc.j] + 1 && res_cap(arc) > 0) {
	        return arc;
        }
    }
    return NULL;
}

void SubmodularFlow::Push(Arc arc) {
    REAL delta;

    if (arc.j == s) {  // reverse arc
        delta = std::min(excess[arc.i], m_phi_si[arc.i]);
        m_phi_si[arc.i] -= delta;
    } else if (arc.j == t) {
        delta = std::min(excess[arc.i], m_c_it[arc.i] - m_phi_it[arc.i]);
        m_phi_it[arc.i] += delta;
    } else { // Clique arc
        delta = std::min(excess[arc.i], Clique::CliqueResidualCapacity(arc));
        std::vector<REAL> & alpha_ci = m_cliques[arc.c].AlphaCi();
        alpha_ci[arc.i] += delta
        alpha_ci[arc.j] -= delta
    }
    // Update (residual capacities) and excesses
    excess[arc.i] -= delta;
    excess[arc.j] += delta;
}

void SubmodularFlow::Relabel(NodeId i) {
    dis[i] = m_num_nodes + 5;  // init to > max
    for(std::vector<Arc>::iterator it = m_arc_list[i].begin();
            it != m_arc_list[i].end(); ++it) {
        Arc arc = *it;
        dis[i] = std::min (dis[i], dis[arc.j] + 1);
    }
}

/* Calculates minimum f_bar w.r.t. to a Clique */
REAL Clique::CliqueResidualCapacity(Arc arc) {
    Clique clique = m_cliques[arc.c];
    const size_t u_idx = std::find(clique.m_nodes.begin(), clique.m_nodes.end(), arc.i) - clique.m_nodes.begin();
    const size_t v_idx = std::find(clique.m_nodes.begin(), clique.m_nodes.end(), arc.j) - clique.m_nodes.end();

    REAL min_f_bar = std::numeric_limits<REAL>::max();
    Assignment num_assgns = 1 << clique.m_nodes.size();
    for (Assignment assgn = 0; assgn < num_assgns; ++assgn) {
        if (assgn & (1 << u_idx) && !(assgn & (1 << v_idx))) {
            // then assgn is a set separating u from v
            f_bar = m_energy[assgn] - sum_alpha_ci;
	        if (f_bar < min_f_bar) min_f_bar = f_bar;
        }
    }
    return min_f_bar;
}

////////    end of push relabel stuff /////////////////

void SubmodularFlow::ComputeMinCut() {
    // Implement me (Sam)
}

REAL SubmodularFlow::ComputeEnergy() const {
    REAL total = 0;
    for (NodeId i = 0; i < m_num_nodes; ++i) {
        if (m_labels[i] == 1) total += m_c_it;
        else total += m_c_si;
    }
    for (const CliquePtr& cp : m_cliques) {
        total += cp->ComputeEnergy(m_labels);
    }
    return total;
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
