#include <iostream>
#include <fstream>
#include "submodular-ibfs.hpp"
#include <algorithm>
#include <limits>
#include <queue>
#include <chrono>

typedef std::chrono::system_clock::time_point TimePt;
typedef std::chrono::duration<double> Duration;
typedef std::chrono::system_clock Clock;

SubmodularIBFS::SubmodularIBFS()
    : s(-1), 
    t(-1),
    m_constant_term(0),
    m_num_nodes(0),
    m_nodes(),
    m_c_si(),
    m_c_it(),
    m_phi_si(),
    m_phi_it(),
    m_labels(),
    m_num_cliques(0),
    m_cliques(),
    m_neighbors(),
    m_num_clique_pushes(0)
{ }

SubmodularIBFS::NodeId SubmodularIBFS::AddNode(int n) {
    ASSERT(n >= 1);
    ASSERT(s == -1);
    NodeId first_node = m_num_nodes;
    for (int i = 0; i < n; ++i) {
        m_nodes.push_back(Node());
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

int SubmodularIBFS::GetLabel(NodeId n) const {
    return m_labels[n];
}

void SubmodularIBFS::AddUnaryTerm(NodeId n, REAL E0, REAL E1) {
    // Reparametize so that E0, E1 >= 0
    if (E0 < 0) {
        AddConstantTerm(E0);
        E1 -= E0;
        E0 = 0;
    }
    if (E1 < 0) {
        AddConstantTerm(E1);
        E0 -= E1;
        E1 = 0;
    }
    m_c_si[n] += E0;
    m_c_it[n] += E1;
}

void SubmodularIBFS::AddUnaryTerm(NodeId n, REAL coeff) {
    AddUnaryTerm(n, 0, coeff);
}

void SubmodularIBFS::ClearUnaries() {
    for (NodeId i = 0; i < m_num_nodes; ++i) {
        m_c_si[i] = m_c_it[i] = 0;
        m_phi_si[i] = m_phi_it[i] = 0;
    }
}

void SubmodularIBFS::AddClique(const std::vector<NodeId>& nodes, const std::vector<REAL>& energyTable, bool normalize) {
    ASSERT(s == -1);
    m_cliques.emplace_back(nodes, energyTable);
    for (NodeId i : nodes) {
        ASSERT(0 <= i && i < m_num_nodes);
        m_neighbors[i].push_back(m_num_cliques);
    }
    if (normalize)
        m_cliques[m_num_cliques].NormalizeEnergy(*this);//Chen
    m_num_cliques++;
}

void SubmodularIBFS::AddPairwiseTerm(NodeId i, NodeId j, REAL E00, REAL E01, REAL E10, REAL E11) {
    ASSERT(s == -1);
    std::vector<NodeId> nodes{i, j};
    std::vector<REAL> energyTable{E00, E01, E10, E11};
    AddClique(nodes, energyTable);
}

void SubmodularIBFS::GraphInit()
{
    // Check to see if we've already initialized, if so: do nothing
    if (s != -1) return;

    auto start = Clock::now();

    // super source and sink
    s = m_num_nodes; t = m_num_nodes + 1;
    num_edges = 2 * m_num_nodes; // source sink edges
    m_nodes.push_back(Node());
    m_nodes.push_back(Node());

    // initialize arc lists
    Arc arc;
    arc.i_idx = arc.j_idx = 0;
    for (NodeId i = 0; i < m_num_nodes; ++i) {
        // arcs from source
        arc.i = s;
        arc.j = i;
        arc.c = -1;
        m_nodes[s].out_arcs.push_back(arc);
        m_nodes[i].in_arcs.push_back(arc);

        // arcs to source
        arc.i = i;
        arc.j = s;
        m_nodes[arc.i].out_arcs.push_back(arc);
        m_nodes[s].in_arcs.push_back(arc);

        // arcs to sink
        arc.j = t;
        m_nodes[arc.i].out_arcs.push_back(arc);
        m_nodes[t].in_arcs.push_back(arc);

        // arcs from sink
        arc.i = t;
        arc.j = i;
        m_nodes[t].out_arcs.push_back(arc);
        m_nodes[i].in_arcs.push_back(arc);
    }

    // Arcs between nodes of clique
    for (int cid = 0; cid < m_num_cliques; ++cid) {
        auto& c = m_cliques[cid];
        int size = c.Nodes().size();
        num_edges += size*(size-1);
        for (NodeId i : c.Nodes()) {
            for (NodeId j : c.Nodes()) {
                if (i == j) continue;
                arc.i = i;
                arc.j = j;
                arc.c = cid;
                arc.i_idx = c.GetIndex(i);
                arc.j_idx = c.GetIndex(j);
                m_nodes[i].out_arcs.push_back(arc);
                m_nodes[j].in_arcs.push_back(arc);
            }
        }
    }
    
    // Sort the arc lists, to ensure consistency of current-arc heuristic
    for (NodeId i = 0; i < m_num_nodes; ++i) {
        std::sort(m_nodes[i].out_arcs.begin(), m_nodes[i].out_arcs.end(),
                [](const Arc& n1, const Arc& n2) { return n1.j < n2.j || (n1.j == n2.j && n1.c < n2.c); });
        std::sort(m_nodes[i].in_arcs.begin(), m_nodes[i].in_arcs.end(),
                [](const Arc& n1, const Arc& n2) { return n1.i < n2.i || (n1.i == n2.i && n1.c < n2.c); });
    }
    m_graphInitTime += Duration{ Clock::now() - start }.count();
}

void SubmodularIBFS::IBFSInit()
{
    auto start = Clock::now();
    // reset distance, state and parent
    for (int i = 0; i < m_num_nodes + 2; ++i) {
        Node& node = m_nodes[i];
        node.dis = std::numeric_limits<int>::max();
        node.state = NodeState::N;
        node.parent = i;
    }
    m_source_layers = std::vector<NodeQueue>(m_num_nodes);
    m_sink_layers = std::vector<NodeQueue>(m_num_nodes);

    m_source_orphans.clear();
    m_sink_orphans.clear();

    auto& sNode = m_nodes[s];
    sNode.state = NodeState::S;
    sNode.dis = 0;
    m_source_layers[0].push_back(sNode);
    auto& tNode = m_nodes[t];
    tNode.state = NodeState::T;
    tNode.dis = 0;
    m_sink_layers[0].push_back(tNode);

    // Reset Clique parameters
    for (int cid = 0; cid < m_num_cliques; ++cid) {
        auto& c = m_cliques[cid];
        c.ResetAlpha();
        c.ComputeMinTightSets();
    }

    // saturate all s-i-t paths
    for (NodeId i = 0; i < m_num_nodes; ++i) {
        REAL min_cap = std::min(m_c_si[i], m_c_it[i]);
        m_phi_si[i] = min_cap;
        m_phi_it[i] = min_cap;
        if (m_c_si[i] > min_cap) {
            auto& node = m_nodes[i];
            node.state = NodeState::S;
            node.dis = 1;
            AddToLayer(i);
            node.parent_arc = std::find_if(node.in_arcs.begin(), node.in_arcs.end(),
                    [&](const Arc& n) { return n.i == s; });
            node.parent = s;
            ASSERT(node.parent_arc->j == i);
            ASSERT(NonzeroCap(*node.parent_arc));
        } else if (m_c_it[i] > min_cap) {
            auto& node = m_nodes[i];
            node.state = NodeState::T;
            node.dis = 1;
            AddToLayer(i);
            node.parent_arc = std::find_if(node.out_arcs.begin(), node.out_arcs.end(),
                    [&](const Arc& n) { return n.j == t; });
            node.parent = t;
            ASSERT(node.parent_arc->i == i);
            ASSERT(NonzeroCap(*node.parent_arc));
        }
    }
    m_initTime += Duration{ Clock::now() - start }.count();
}

void SubmodularIBFS::IBFS() {
    auto start = Clock::now();
    GraphInit();
    IBFSInit();

    // Set up initial current_q and search nodes to make it look like
    // we just finished scanning the sink node
    NodeQueue* current_q = &(m_sink_layers[0]);
    m_search_node_iter = current_q->end();
    m_search_node_end = current_q->end();

    m_forward_search = false;
    m_source_tree_d = 1;
    m_sink_tree_d = 0;

    while (!current_q->empty()) {
        if (m_search_node_iter == m_search_node_end) {
            // Swap queues and continue
            if (m_forward_search) {
                m_source_tree_d++;
                current_q = &(m_sink_layers[m_sink_tree_d]);
            } else {
                m_sink_tree_d++;
                current_q = &(m_source_layers[m_source_tree_d]);
            }
            m_search_node_iter = current_q->begin();
            m_search_node_end = current_q->end();
            m_forward_search = !m_forward_search;
            if (!current_q->empty()) {
                Node& n = *m_search_node_iter;
                if (m_forward_search) {
                    ASSERT(n.state == NodeState::S || n.state == NodeState::S_orphan);
                    m_search_arc = n.out_arcs.begin();
                    m_search_arc_end = n.out_arcs.end();
                } else {
                    ASSERT(n.state == NodeState::T || n.state == NodeState::T_orphan);
                    m_search_arc = n.in_arcs.begin();
                    m_search_arc_end = n.in_arcs.end();
                }
            }
            continue;
        }
        Node& n = *m_search_node_iter;
        NodeId search_node = &n - m_nodes.data();
        int distance;
        if (m_forward_search) {
            distance = m_source_tree_d;
        } else {
            distance = m_sink_tree_d;
        }
        ASSERT(n.dis == distance);
        // Advance m_search_arc until we find a residual arc
        while (m_search_arc != m_search_arc_end && !NonzeroCap(*m_search_arc))
            m_search_arc++;

        if (m_search_arc != m_search_arc_end) {
            NodeId neighbor;
            if (m_forward_search) neighbor = m_search_arc->j;
            else neighbor = m_search_arc->i;
            NodeState neighbor_state = m_nodes[neighbor].state;
            if (neighbor_state == n.state) {
                ASSERT(m_nodes[neighbor].dis <= n.dis + 1);
                if (m_nodes[neighbor].dis == n.dis+1) {
                    if (n.state == NodeState::S
                            && (search_node < m_nodes[neighbor].parent
                                || (search_node == m_nodes[neighbor].parent
                                    && m_search_arc->c < m_nodes[neighbor].parent_arc->c))) {
                        m_nodes[neighbor].parent_arc = std::find_if(m_nodes[neighbor].in_arcs.begin(),
                                m_nodes[neighbor].in_arcs.end(),
                                [&](const Arc& a) { return a.i == search_node && a.c == m_search_arc->c; });
                        m_nodes[neighbor].parent = search_node;
                        ASSERT(NonzeroCap(*m_nodes[neighbor].parent_arc));
                    } else if (n.state == NodeState::T
                            && (search_node < m_nodes[neighbor].parent
                                || (search_node == m_nodes[neighbor].parent
                                    && m_search_arc->c < m_nodes[neighbor].parent_arc->c))) {
                        m_nodes[neighbor].parent_arc = std::find_if(m_nodes[neighbor].out_arcs.begin(),
                                m_nodes[neighbor].out_arcs.end(),
                                [&](const Arc& a) { return a.j == search_node && a.c == m_search_arc->c; });
                        m_nodes[neighbor].parent = search_node;
                        ASSERT(NonzeroCap(*m_nodes[neighbor].parent_arc));
                    }
                }
                m_search_arc++;
            } else if (neighbor_state == NodeState::N) {
                // Then we found an unlabeled node, add it to the tree
                m_nodes[neighbor].state = n.state;
                m_nodes[neighbor].dis = n.dis + 1;
                AddToLayer(neighbor);
                if (m_forward_search) {
                    m_nodes[neighbor].parent_arc = std::find_if(m_nodes[neighbor].in_arcs.begin(),
                            m_nodes[neighbor].in_arcs.end(),
                            [&](const Arc& a) { return a.i == search_node && a.c == m_search_arc->c; });
                    ASSERT(m_nodes[neighbor].parent_arc != m_nodes[neighbor].in_arcs.end());
                    ASSERT(NonzeroCap(*m_nodes[neighbor].parent_arc));
                    m_nodes[neighbor].parent = search_node;
                } else {
                    m_nodes[neighbor].parent_arc = std::find_if(m_nodes[neighbor].out_arcs.begin(),
                            m_nodes[neighbor].out_arcs.end(),
                            [&](const Arc& a) { return a.j == search_node && a.c == m_search_arc->c; });
                    ASSERT(m_nodes[neighbor].parent_arc != m_nodes[neighbor].out_arcs.end());
                    ASSERT(NonzeroCap(*m_nodes[neighbor].parent_arc));
                    m_nodes[neighbor].parent = search_node;
                }
                m_search_arc++;
            } else {
                // Then we found an arc to the other tree
                ASSERT(neighbor_state != NodeState::S_orphan && neighbor_state != NodeState::T_orphan);
                ASSERT(NonzeroCap(*m_search_arc));
                Augment(*m_search_arc);
                Adopt();
            }
        } else {
            // No more arcs to scan from this node, so remove from queue
            AdvanceSearchNode();
        }
    } // End while
    m_totalTime += Duration{ Clock::now() - start }.count();

    std::cout << "Total time:      " << m_totalTime << "\n";
    std::cout << "Graph init time: " << m_graphInitTime << "\n";
    std::cout << "Init time:       " << m_initTime << "\n";
    std::cout << "Augment time:    " << m_augmentTime << "\n";
    std::cout << "Adopt time:      " << m_adoptTime << "\n";
}

void SubmodularIBFS::Augment(Arc& arc) {
    auto start = Clock::now();

    NodeId i = arc.i;
    NodeId j = arc.j;
    REAL bottleneck = ResCap(arc);
    NodeId current = i;
    while (current != s) {
        ASSERT(m_nodes[current].state == NodeState::S);
        Arc& a = *m_nodes[current].parent_arc;
        bottleneck = std::min(bottleneck, ResCap(a));
        current = a.i;
    }
    current = j;
    while (current != t) {
        ASSERT(m_nodes[current].state == NodeState::T);
        Arc& a = *m_nodes[current].parent_arc;
        bottleneck = std::min(bottleneck, ResCap(a));
        current = a.j;
    }
    ASSERT(bottleneck > 0);
    //ASSERT(bottleneck > -1e-7);//Chen
    Push(arc, bottleneck);
    current = i;
    while (current != s) {
        Arc& a = *m_nodes[current].parent_arc;
        Push(a, bottleneck);
        current = a.i;
    }
    current = j;
    while (current != t) {
        Arc& a = *m_nodes[current].parent_arc;
        Push(a, bottleneck);
        current = a.j;
    }
    m_augmentTime += Duration{ Clock::now() - start }.count();
}

void SubmodularIBFS::Adopt() {
    auto start = Clock::now();
    while (!m_source_orphans.empty()) {
        NodeId i = m_source_orphans.front();
        Node& n = m_nodes[i];
        m_source_orphans.pop_front();
        int old_dist = n.dis;
        while (n.parent_arc != n.in_arcs.end()
                && (m_nodes[n.parent].state == NodeState::T
                    || m_nodes[n.parent].state == NodeState::T_orphan
                    || m_nodes[n.parent].state == NodeState::N
                    || m_nodes[n.parent].dis != old_dist - 1
                    || !NonzeroCap(*n.parent_arc))) {
            n.parent_arc++;
            n.parent = n.parent_arc->i;
        }
        if (n.parent_arc == n.in_arcs.end()) {
            RemoveFromLayer(i);
            // We didn't find a new parent with the same label, so do a relabel
            n.dis = std::numeric_limits<int>::max()-1;
            for (auto iter = n.in_arcs.begin(); iter != n.in_arcs.end(); ++iter) {
                if (m_nodes[iter->i].dis < n.dis
                        && (m_nodes[iter->i].state == NodeState::S
                            || m_nodes[iter->i].state == NodeState::S_orphan)
                        && NonzeroCap(*iter)) {
                    n.dis = m_nodes[iter->i].dis;
                    n.parent_arc = iter;
                    ASSERT(NonzeroCap(*n.parent_arc));
                    n.parent = iter->i;
                }
            }
            n.dis++;
            int cutoff_distance = m_source_tree_d;
            if (m_forward_search) cutoff_distance += 1;
            if (n.dis > cutoff_distance) {
                n.state = NodeState::N;
            } else {
                n.state = NodeState::S;
                AddToLayer(i);
            }
            if (n.dis > old_dist) {
                for (Arc& a : n.out_arcs) {
                    if (m_nodes[a.j].parent == i)
                        MakeOrphan(a.j);
                }
            }
        } else {
            ASSERT(NonzeroCap(*n.parent_arc));
            n.state = NodeState::S;
        }
    }
    while (!m_sink_orphans.empty()) {
        NodeId i = m_sink_orphans.front();
        Node& n = m_nodes[i];
        m_sink_orphans.pop_front();
        int old_dist = n.dis;
        while (n.parent_arc != n.out_arcs.end()
                && (m_nodes[n.parent].state == NodeState::S
                    || m_nodes[n.parent].state == NodeState::S_orphan
                    || m_nodes[n.parent].state == NodeState::N
                    || m_nodes[n.parent].dis != old_dist - 1
                    || !NonzeroCap(*n.parent_arc))) {
            n.parent_arc++;
            n.parent = n.parent_arc->j;
        }
        if (n.parent_arc == n.out_arcs.end()) {
            RemoveFromLayer(i);
            // We didn't find a new parent with the same label, so do a relabel
            n.dis = std::numeric_limits<int>::max()-1;
            for (auto iter = n.out_arcs.begin(); iter != n.out_arcs.end(); ++iter) {
                if (m_nodes[iter->j].dis < n.dis
                        && (m_nodes[iter->j].state == NodeState::T
                            || m_nodes[iter->j].state == NodeState::T_orphan)
                        && NonzeroCap(*iter)) {
                    n.dis = m_nodes[iter->j].dis;
                    n.parent_arc = iter;
                    ASSERT(NonzeroCap(*n.parent_arc));
                    n.parent = iter->j;
                }
            }
            n.dis++;
            int cutoff_distance = m_sink_tree_d;
            if (!m_forward_search) cutoff_distance += 1;
            if (n.dis > cutoff_distance) {
                n.state = NodeState::N;
            } else {
                n.state = NodeState::T;
                AddToLayer(i);
            }
            if (n.dis > old_dist) {
                for (Arc& a : n.in_arcs) {
                    if (m_nodes[a.i].parent == i)
                        MakeOrphan(a.i);
                }
            }
        } else {
            ASSERT(NonzeroCap(*n.parent_arc));
            n.state = NodeState::T;
        }
    }
    m_adoptTime += Duration{ Clock::now() - start }.count();
}

void SubmodularIBFS::MakeOrphan(NodeId i) {
    Node& n = m_nodes[i];
    if (n.state == NodeState::S) {
        n.state = NodeState::S_orphan;
        m_source_orphans.push_back(i);
    } else if (n.state == NodeState::T) {
        n.state = NodeState::T_orphan;
        m_sink_orphans.push_back(i);
    }
}


REAL SubmodularIBFS::ResCap(Arc& arc) {
    if (arc.c >= 0) {
        return m_cliques[arc.c].ExchangeCapacity(arc.i_idx, arc.j_idx);
    } else if (arc.i == s) {
        return m_c_si[arc.j] - m_phi_si[arc.j];
    } else if (arc.j == s) {
        return m_phi_si[arc.i];
    } else if (arc.i == t) {
        return m_phi_it[arc.j];
    } else if (arc.j == t) {
        return m_c_it[arc.i] - m_phi_it[arc.i];
    } else {
        ASSERT(false /* should not reach here */);
    }
}

bool SubmodularIBFS::NonzeroCap(Arc& arc) {
    if (arc.c >= 0) {
        return m_cliques[arc.c].NonzeroCapacity(arc.i_idx, arc.j_idx);
    } else if (arc.i == s) {
        return (m_c_si[arc.j] - m_phi_si[arc.j]) != 0;
    } else if (arc.j == s) {
        return m_phi_si[arc.i] != 0;
    } else if (arc.i == t) {
        return m_phi_it[arc.j] != 0;
    } else if (arc.j == t) {
        return (m_c_it[arc.i] - m_phi_it[arc.i]) != 0;
    } else {
        ASSERT(false /* should not reach here */);
    }
}

void SubmodularIBFS::Push(Arc& arc, REAL delta) {
    ASSERT(delta > 0);
    //ASSERT(delta > -1e-7);//Chen
    ASSERT(arc.j != s && arc.i != t);
    if (arc.i == s) { // reverse arc
        //ASSERT(delta <= m_c_si[arc.j] - m_phi_si[arc.j] + 1e-7);//Chen
        ASSERT(delta <= m_c_si[arc.j] - m_phi_si[arc.j]);
        m_phi_si[arc.j] += delta;
        if (m_phi_si[arc.j] == m_c_si[arc.j]) {
            MakeOrphan(arc.j);
        }
    } else if (arc.j == t) {
        //ASSERT(delta <= m_c_it[arc.i] - m_phi_it[arc.i] + 1e-7);//Chen
        ASSERT(delta <= m_c_it[arc.i] - m_phi_it[arc.i]);
        m_phi_it[arc.i] += delta;
        if (m_phi_it[arc.i] == m_c_it[arc.i]) {
            MakeOrphan(arc.i);
        }
    } else { // Clique arc
        m_num_clique_pushes++;
        //std::cout << "Pushing on clique arc (" << arc.i << ", " << arc.j << ") -- delta = " << delta << std::endl;
        auto& c = m_cliques[arc.c];
        c.Push(arc.i_idx, arc.j_idx, delta);
        c.Time()++;
        for (NodeId n : c.Nodes()) {
            if (m_nodes[n].state == NodeState::N)
                continue;
            Arc& parent_arc = *m_nodes[n].parent_arc;
            if (parent_arc.c == arc.c && !NonzeroCap(parent_arc)) {
                MakeOrphan(n);
            }
        }
    }
}

/////////////// end of push relabel ///////////////////


void SubmodularIBFS::ComputeMinCut() {
    for (NodeId i = 0; i < m_num_nodes; ++i) {
        if (m_nodes[i].state == NodeState::T)
            m_labels[i] = 0;
        else if (m_nodes[i].state == NodeState::S)
            m_labels[i] = 1;
        else {
            ASSERT(m_nodes[i].state == NodeState::N);
            // Put N nodes on whichever side could still grow
            m_labels[i] = !m_forward_search;
        }
    }
}

void SubmodularIBFS::Solve() {
    IBFS();
    ComputeMinCut();
}

REAL SubmodularIBFS::ComputeEnergy() const {
    return ComputeEnergy(m_labels);
}

REAL SubmodularIBFS::ComputeEnergy(const std::vector<int>& labels) const {
    REAL total = m_constant_term;
    for (NodeId i = 0; i < m_num_nodes; ++i) {
        if (labels[i] == 1) total += m_c_it[i];
        else total += m_c_si[i];
    }
    for (const auto& c : m_cliques) {
        total += c.ComputeEnergy(labels);
    }
    return total;
}

static void CheckSubmodular(size_t n, const std::vector<REAL>& m_energy) {
    typedef int32_t Assignment;
    Assignment max_assgn = 1 << n;
    for (Assignment s = 0; s < max_assgn; ++s) {
        for (size_t i = 0; i < n; ++i) {
            Assignment si = s | (1 << i);
            if (si != s) {
                for (size_t j = i+1; j < n; ++j) {
                    Assignment t = s | (1 << j);
                    if (t != s && j != i) {
                        Assignment ti = t | (1 << i);
                        // Decreasing marginal costs, so we require
                        // f(ti) - f(t) <= f(si) - f(s)
                        // i.e. f(si) - f(s) - f(ti) + f(t) >= 0
                        REAL violation = -m_energy[si] - m_energy[t]
                            + m_energy[s] + m_energy[ti];
                        if (violation > 0) std::cout << violation << std::endl;
                        ASSERT(violation <= 0);
                    }
                }
            }
        }
    }
}

void SubmodularIBFS::IBFSEnergyTableClique::NormalizeEnergy(SubmodularIBFS& sf) {
    EnforceSubmodularity();
    const size_t n = this->m_nodes.size();
    CheckSubmodular(n, m_energy);
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
        ASSERT(m_energy[a] >= 0);
        /* FIXME: the above only works if the energy is actually submodular
* not epsilon-submodular. To make everything positive even if not,
* we truncate to zero.
*/
        //m_energy[a] = std::max(0, m_energy[a]);
        m_alpha_energy[a] = m_energy[a];
    }
    ComputeMinTightSets();

    sf.AddConstantTerm(constant_term);
    for (size_t i = 0; i < n; ++i) {
        sf.AddUnaryTerm(this->m_nodes[i], -marginals[i], 0);
    }

    CheckSubmodular(n, m_energy);

}

REAL SubmodularIBFS::IBFSEnergyTableClique::ComputeEnergy(const std::vector<int>& labels) const {
    Assignment assgn = 0;
    for (size_t i = 0; i < this->m_nodes.size(); ++i) {
        NodeId n = this->m_nodes[i];
        if (labels[n] == 1) {
            assgn |= 1 << i;
        }
    }
    return m_energy[assgn];
}

REAL SubmodularIBFS::IBFSEnergyTableClique::ExchangeCapacity(size_t u_idx, size_t v_idx) const {
    const size_t n = this->m_nodes.size();
    ASSERT(u_idx < n);
    ASSERT(v_idx < n);

    REAL min_energy = std::numeric_limits<REAL>::max();
    Assignment num_assgns = 1 << n;
    const Assignment bound = num_assgns-1;
    const Assignment u_mask = 1 << u_idx;
    const Assignment v_mask = 1 << v_idx;
    const Assignment uv_mask = u_mask | v_mask;
    const Assignment subset_mask = bound & ~uv_mask;
    // Terrible bit-hacks to optimize the living hell out of this function
    // Iterate over all assignments without u_idx or v_idx set
    Assignment assgn = subset_mask;
    do {
        Assignment u_sep = assgn | u_mask;
        REAL energy = m_alpha_energy[u_sep];
        if (energy < min_energy) min_energy = energy;
        assgn = ((assgn - 1) & subset_mask);
    } while (assgn != subset_mask);

    return min_energy;
}

void SubmodularIBFS::IBFSEnergyTableClique::Push(size_t u_idx, size_t v_idx, REAL delta) {
    ASSERT(u_idx >= 0 && u_idx < this->m_nodes.size());
    ASSERT(v_idx >= 0 && v_idx < this->m_nodes.size());
    m_alpha_Ci[u_idx] += delta;
    m_alpha_Ci[v_idx] -= delta;
    const size_t n = this->m_nodes.size();
    Assignment num_assgns = 1 << n;
    const Assignment bound = num_assgns-1;
    const Assignment u_mask = 1 << u_idx;
    const Assignment v_mask = 1 << v_idx;
    const Assignment uv_mask = u_mask | v_mask;
    const Assignment subset_mask = bound & ~uv_mask;
    // Terrible bit-hacks to optimize the living hell out of this function
    // Iterate over all assignments without u_idx or v_idx set
    Assignment assgn = subset_mask;
    do {
        Assignment u_sep = assgn | u_mask;
        Assignment v_sep = assgn | v_mask;
        m_alpha_energy[u_sep] -= delta;
        m_alpha_energy[v_sep] += delta;
        assgn = ((assgn - 1) & subset_mask);
    } while (assgn != subset_mask);

    ComputeMinTightSets();
}

void SubmodularIBFS::IBFSEnergyTableClique::ComputeMinTightSets() {
    size_t n = this->m_nodes.size();
    Assignment num_assgns = 1 << n;
    const Assignment bound = num_assgns-1;
    for (auto& a : m_min_tight_set)
        a = bound;
    for (Assignment assgn = bound-1; assgn >= 1; --assgn) {
        if (m_alpha_energy[assgn] == 0) {
            for (size_t i = 0; i < n; ++i) {
                //ASSERT(m_alpha_energy[m_min_tight_set[i] & assgn] == 0);
                //ASSERT(m_alpha_energy[m_min_tight_set[i] | assgn] == 0);
                if ((assgn & (1 << i)) != 0)
                    m_min_tight_set[i] = assgn;
            }
        }
    }
}

bool SubmodularIBFS::IBFSEnergyTableClique::NonzeroCapacity(size_t u_idx, size_t v_idx) const {
    Assignment min_set = m_min_tight_set[u_idx];
    return (min_set & (1 << v_idx)) != 0;
}

static inline int32_t NextPerm(uint32_t v) {
    uint32_t t = v | (v - 1); // t gets v's least significant 0 bits set to 1
    // Next set to 1 the most significant bit to change,
    // set to 0 the least significant ones, and add the necessary 1 bits.
    return (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(v) + 1));
}

void SubmodularIBFS::IBFSEnergyTableClique::EnforceSubmodularity() {
    // Decreasing marginal costs, so we require
    // f(ti) - f(t) <= f(si) - f(s)
    // i.e. f(si) - f(s) - f(ti) + f(t) >= 0
    // must hold for all subsets s, and t where t = s+j,
    // si = s+i, ti = t+i
    const size_t n = this->m_nodes.size();
    Assignment max_assgn = 1 << n;
    // Need to iterate over all k bit subsets in increasing k
    for (size_t k = 0; k < n; ++k) {
        Assignment bound;
        if (k == 0) bound = 0;
        else bound = max_assgn - 1;
        Assignment s = (1 << k) - 1;
        do {
            for (size_t i = 0; i < n; ++i) {
                Assignment si = s | (1 << i);
                if (si != s) {
                    for (size_t j = i+1; j < n; ++j) {
                        Assignment t = s | (1 << j);
                        if (t != s && j != i) {
                            Assignment ti = t | (1 << i);
                            REAL violation = -m_energy[si] - m_energy[t]
                                + m_energy[s] + m_energy[ti];
                            if (violation > 0) {
                                m_energy[ti] -= violation;
                            }
                        }
                    }
                }
            }
            s = NextPerm(s);
        } while (s < bound);
    }
}

void SubmodularIBFS::IBFSEnergyTableClique::ResetAlpha() {
    for (auto& a : this->m_alpha_Ci) {
        a = 0;
    }
    const size_t n = this->m_nodes.size();
    const Assignment num_assignments = 1 << n;
    for (Assignment a = 0; a < num_assignments; ++a) {
        m_alpha_energy[a] = m_energy[a];
    }
    this->Time()++;
}

void SubmodularIBFS::AddToLayer(NodeId i) {
    auto& node = m_nodes[i];
    int dis = node.dis;
    if (node.state == NodeState::S) {
        m_source_layers[dis].push_front(node);
    } else if (node.state == NodeState::T) {
        m_sink_layers[dis].push_front(node);
    } else {
        ASSERT(false);
    }
}

void SubmodularIBFS::RemoveFromLayer(NodeId i) {
    auto& node = m_nodes[i];
    if (m_search_node_iter != m_search_node_end && &(*m_search_node_iter) == &node)
        AdvanceSearchNode();
    int dis = node.dis;
    if (node.state == NodeState::S || node.state == NodeState::S_orphan) {
        auto& layer = m_source_layers[dis];
        layer.erase(layer.iterator_to(node));
    } else if (node.state == NodeState::T || node.state == NodeState::T_orphan) {
        auto& layer = m_sink_layers[dis];
        layer.erase(layer.iterator_to(node));
    } else {
        ASSERT(false);
    }
}

void SubmodularIBFS::AdvanceSearchNode() {
    m_search_node_iter++;
    if (m_search_node_iter != m_search_node_end) {
        Node& n = *m_search_node_iter;
        if (m_forward_search) {
            ASSERT(n.state == NodeState::S || n.state == NodeState::S_orphan);
            m_search_arc = n.out_arcs.begin();
            m_search_arc_end = n.out_arcs.end();
        } else {
            ASSERT(n.state == NodeState::T || n.state == NodeState::T_orphan);
            m_search_arc = n.in_arcs.begin();
            m_search_arc_end = n.in_arcs.end();
        }
    }
}
