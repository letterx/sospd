#include "spd1.hpp"
#include "clique.hpp"

SubmodularPrimalDual1::SubmodularPrimalDual1(Label max_label)
    : m_num_labels(max_label),
    m_constant_term(0),
    m_cliques(),
    m_unary_cost(),
    m_labels()
{ }

SubmodularPrimalDual1::NodeId SubmodularPrimalDual1::AddNode(int n) {
    NodeId ret = m_labels.size();
    for (int i = 0; i < n; ++i) {
        m_labels.push_back(0);
        UnaryCost uc(m_num_labels, 0);
        m_unary_cost.push_back(uc);
    }
    return ret;
}

int SubmodularPrimalDual1::GetLabel(NodeId i) const {
    return m_labels[i];
}

void SubmodularPrimalDual1::AddConstantTerm(REAL c) {
    m_constant_term += c;
}

void SubmodularPrimalDual1::AddUnaryTerm(NodeId i, const std::vector<REAL>& coeffs) {
    ASSERT(coeffs.size() == m_num_labels);
    for (size_t j = 0; j < m_num_labels; ++j) {
        m_unary_cost[i][j] += coeffs[j];
    }
}

void SubmodularPrimalDual1::AddClique(const CliquePtr& cp) {
    m_cliques.push_back(cp);
}

void SubmodularPrimalDual1::InitialLabeling() {
    const NodeId n = m_unary_cost.size();
    for (NodeId i = 0; i < n; ++i) {
        REAL best_cost = std::numeric_limits<REAL>::max();
        for (size_t l = 0; l < m_num_labels; ++l) {
            if (m_unary_cost[i][l] < best_cost) {
                best_cost = m_unary_cost[i][l];
                m_labels[i] = l;
            }
        }
    }
}

void SubmodularPrimalDual1::InitialDual() {
	m_dual.clear();
	std::vector<Label> labelBuf;
	for (const CliquePtr& cp : m_cliques) {
        const Clique& c = *cp;
		std::vector<NodeId> nodes = c.Nodes();
		size_t k = nodes.size();
	    std::cout.flush();
		std::vector<size_t> cnt(m_num_labels, 0);
		labelBuf.clear();
		for (size_t i = 0; i < k; ++i) {
			labelBuf.push_back(m_labels[nodes[i]]);
			cnt[m_labels[nodes[i]]]++;
		}
		REAL energy = c.m_f_min;
		Dual newDual;
		newDual.clear();
		for (size_t i = 0; i < k; ++i) {
			std::vector<REAL> dual(m_num_labels, 0);
			newDual.push_back(dual);
		}
		for (size_t i = 0; i < m_num_labels; ++i){
			if (cnt[i] != 0) {
				for (size_t j = 0; j < k; ++j) {
					if (labelBuf[j] == i) {
						newDual[j][i] = energy / k;
					}
					else {
						ASSERT(k != cnt[i]);
						newDual[j][i] = -energy / k * cnt[i] / (k - cnt[i]);
					}
				}
			}
		}
		m_dual.push_back(newDual);
    }
}

void SubmodularPrimalDual1::InitialNodeCliqueList() {
    size_t n = m_labels.size();
    m_node_clique_list.clear();
    for (size_t i = 0; i < n; ++i) {
        std::vector<std::pair<size_t, size_t> > list;
        m_node_clique_list.push_back(list);
    }
    int clique_index = 0;
    for (const CliquePtr& cp : m_cliques) {
        const Clique& c = *cp;
        std::vector<NodeId> nodes = c.Nodes();
        const size_t k = nodes.size();
        for (size_t i = 0; i < k; ++i) {
            m_node_clique_list[nodes[i]].push_back(std::make_pair(clique_index, i));
        }
        ++clique_index;
    }
}

REAL SubmodularPrimalDual1::ComputeHeight(NodeId i, Label x) {
    REAL ret = m_unary_cost[i][x];
    for (size_t j = 0; j < m_node_clique_list[i].size(); ++j) {
        size_t cliqueId = m_node_clique_list[i][j].first;
        size_t nodeId = m_node_clique_list[i][j].second;
        ret += m_dual[cliqueId][nodeId][x];
    }
    return ret;
}

void SubmodularPrimalDual1::SetupAlphaEnergy(Label alpha, IBFSGraph<flowtype, flowtype, flowtype>& g) {
    //TODO
    typedef int32_t Assgn;
    const size_t n = m_labels.size();
    crf.AddNode(n);
    for (size_t i = 0; i < n; ++i) {
        REAL height_x = ComputeHeight(i, m_labels[i]);
        REAL height_alpha = ComputeHeight(i, alpha);
        if (height_x > height_alpha) {
            crf.AddUnaryTerm(i, height_x - height_alpha, 0);
        }
        else {
            crf.AddUnaryTerm(i, 0, height_alpha - height_x);
        }
    }
    size_t clique_index = 0;
    for (const CliquePtr& cp : m_cliques) {
        const Clique& c = *cp;
        const size_t k = c.Size();
        ASSERT(k < 32);
        const Assgn max_assgn = 1 << k;
        std::vector<REAL> energy_table;
        std::vector<Label> label_buf;
        for (Assgn a = 0; a < max_assgn; ++a) {
            label_buf.clear();
            REAL lambda = 0;
            for (size_t i_idx = 0; i_idx < k; ++i_idx) {
                if (a & (1 << i_idx)) {
                    label_buf.push_back(alpha);
                    lambda += m_dual[clique_index][i_idx][alpha];
                }
                else {
                    Label x = m_labels[c.Nodes()[i_idx]];
                    label_buf.push_back(x);
                    lambda += m_dual[clique_index][i_idx][x];
                }
            }
            energy_table.push_back(c.Energy(label_buf) - lambda);
        }
        crf.AddClique(c.Nodes(), energy_table);
        ++clique_index;
    }
}

bool SubmodularPrimalDual1::UpdatePrimalDual(Label alpha) {
    bool ret = false;
    size_t num_nodes = m_labels.size();
    size_t num_edges = 0;
    for (const CliquePtr& cp : m_cliques) {
        const Clique& c = *cp;
        const size_t k = c.Size();
        num_nodes += k;
        num_edges += k * (k - 1);
    }
    IBFSGraph<flowtype, flowtype, flowtype> g(num_nodes, num_edges);
    SetupAlphaEnergy(alpha, g);
    g.maxFlow();
    size_t n = m_labels.size();
    for (NodeId i = 0; i < n; ++i) {
        int crf_label = g.what_segment(i);
        if (crf_label == 1) {
            if (m_labels[i] != alpha) ret = true;
            m_labels[i] = alpha;
        }
    }
    //TODO
    return ret;
}

void SubmodularPrimalDual1::PostEditDual() {
    std::vector<Label> labelBuf;
    int clique_index = 0;
    for (const CliquePtr& cp : m_cliques) {
        const Clique& c = *cp;
        std::vector<NodeId> nodes = c.Nodes();
        size_t k = nodes.size();
        labelBuf.clear();
        std::vector<size_t> cnt(m_num_labels, 0);
        std::vector<REAL> delta(m_num_labels, 0);
		for (size_t i = 0; i < k; ++i) {
			labelBuf.push_back(m_labels[nodes[i]]);
			cnt[m_labels[nodes[i]]]++;
		}
		REAL energy = 0;
		for (size_t i = 0; i < k; ++i) {
		    energy += m_dual[clique_index][i][labelBuf[i]];
		}
		for (size_t i = 0; i < k; ++i) {
		    delta[labelBuf[i]] += m_dual[clique_index][i][labelBuf[i]] - energy / k;
		    m_dual[clique_index][i][labelBuf[i]] = energy / k;
		}
		for (size_t i = 0; i < k; ++i) {
		    for (Label j = 0; j < m_num_labels; ++j) {
		        if (labelBuf[i] != j) {
		            ASSERT(cnt[j] != k);
		            m_dual[clique_index][i][j] += delta[j] / (k - cnt[j]);
		        }
		    }
		}
		++clique_index;
    }
}

void SubmodularPrimalDual1::Solve() {
	#ifdef PROGRESS_DISPLAY
		std::cout << "(" << std::endl;
		std::cout.flush();
	#endif
	m_num_cliques = m_cliques.size();
	ComputeRho();
	InitialLabeling();
	InitialDual();
	InitialNodeCliqueList();
	#ifdef PROGRESS_DISPLAY
		size_t num_round = 0;
		REAL energy = ComputeEnergy();
		std::cout << "Iteration " << num_round << ": " << energy << std::endl;
		std::cout.flush();
	#endif
	#ifdef DEBUG
	    std::cout << "Init Done!" << std::endl;
        ASSERT(CheckLabelInvariant());
        ASSERT(CheckDualBoundInvariant());
        ASSERT(CheckActiveInvariant());
        ASSERT(CheckZeroSumInvariant());
	#endif
	bool labelChanged = true;
	while (labelChanged){
		labelChanged = false;
		for (size_t alpha = 0; alpha < m_num_labels; ++alpha){
			if (UpdatePrimalDual(alpha)) labelChanged = true;
			PostEditDual();
			#ifdef DEBUG
			    std::cout << "Post-edit Done!" << std::endl;
                ASSERT(CheckLabelInvariant());
                ASSERT(CheckDualBoundInvariant());
                ASSERT(CheckActiveInvariant());
                ASSERT(CheckZeroSumInvariant());
	        #endif
		}
		#ifdef PROGRESS_DISPLAY
			energy = ComputeEnergy();
			num_round++;
			std::cout << "Iteration " << num_round << ": " << energy << std::endl;
			std::cout.flush();
		#endif
	}
	#ifdef DEBUG
	    ASSERT(CheckHeightInvariant());
	#endif
    #ifdef PROGRESS_DISPLAY
	    std::cout << ")" << std::endl;
	    std::cout.flush();
    #endif
}

REAL SubmodularPrimalDual1::ComputeEnergy() const {
    return ComputeEnergy(m_labels);
}

REAL SubmodularPrimalDual1::ComputeEnergy(const std::vector<Label>& labels) const {
    REAL energy = m_constant_term;
    std::vector<Label> labelBuf;
    for (const CliquePtr& cp : m_cliques) {
        const Clique& c = *cp;
        labelBuf.clear();
        for (NodeId i : c.Nodes()) 
            labelBuf.push_back(m_labels[i]);
        energy += c.Energy(labelBuf);
    }
    const NodeId n = m_labels.size();
    for (NodeId i = 0; i < n; ++i)
        energy += m_unary_cost[i][labels[i]];
    return energy;
}
        
void SubmodularPrimalDual1::ComputeRho() {
    m_rho = 1;
    for (const CliquePtr& cp : m_cliques) {
        Clique& c = *cp;
        const size_t k = c.Nodes().size();
        std::vector<Label> labelBuf;
        for (size_t i = 0; i < k; ++i) {
            labelBuf.push_back(0);
        }
        bool t = true;
        REAL max_energy = -1;
        REAL min_energy = -1;
        while (t) {
            REAL energy = c.Energy(labelBuf);
            if (energy > max_energy) max_energy = energy;
            if ((energy > 0) && ((min_energy < 0) || (energy < min_energy))) min_energy = energy;
            labelBuf[0]++;
            for (size_t i = 0; i < k; ++i){
                if (labelBuf[i] == m_num_labels) {
                    if (i + 1 < k) {
                        labelBuf[i] = 0;
                        labelBuf[i + 1]++;
                    }
                    else {
                        t = false;
                    }
                }
                else break;
            }
        }
        c.m_f_max = max_energy;
        c.m_f_min = min_energy;
        if (k * max_energy / min_energy > m_rho) m_rho = k * max_energy / min_energy;
    }
}

double SubmodularPrimalDual1::GetRho() {
    return m_rho;
}

bool SubmodularPrimalDual1::CheckHeightInvariant() {
    size_t m = m_labels.size();
    for (size_t i = 0; i < m; ++i) {
        REAL hx = ComputeHeight(i, m_labels[i]);
        for (Label alpha = 0; alpha < m_num_labels; ++alpha) {
            if (alpha == m_labels[i]) continue;
            REAL halpha = ComputeHeight(i, alpha);
            if (hx > halpha + EPS) {
                std::cout << "Variable: " << i << std::endl;
                std::cout << "Label: " << m_labels[i] << " Height: " << hx << std::endl;
                std::cout << "Label: " << alpha << " Height: " << halpha << std::endl;
                return false;
            }
        }
    }
    return true;
}

bool SubmodularPrimalDual1::CheckLabelInvariant() {
    size_t clique_index = 0;
    for (const CliquePtr& cp : m_cliques) {
        const Clique& c = *cp;
        std::vector<NodeId> nodes = c.Nodes();
        const size_t k = nodes.size();
        std::vector<Label> labelBuf;
        for (size_t i = 0; i < k; ++i) {
            labelBuf.push_back(m_labels[nodes[i]]);
        }
        REAL energy = m_mu * c.Energy(labelBuf);
        REAL sum = 0;
        for (size_t i = 0; i < k; ++i) {
            sum += m_dual[clique_index][i][labelBuf[i]];
        }
        if (fabs(sum - energy) > EPS) {
            std::cout << "CliqueId: " << clique_index << std::endl;
            std::cout << "Energy: " << energy << std::endl;
            std::cout << "DualSum: " << sum << std::endl;
            return false;
        }
        clique_index++;
    }
    return true;
}

bool SubmodularPrimalDual1::CheckDualBoundInvariant() {
    size_t clique_index = 0;
    for (const CliquePtr& cp : m_cliques) {
        const Clique& c = *cp;
        REAL energyBound = c.m_f_max * m_mu;
        for (size_t i = 0; i < m_dual[clique_index].size(); ++i) {
            for (size_t j = 0; j < m_num_labels; ++j) {
                if (m_dual[clique_index][i][j] > energyBound + EPS) {
                    std::cout << "CliqueId: " << clique_index << std::endl;
                    std::cout << "NodeId (w.r.t. Clique): " << i << std::endl;
                    std::cout << "Label: " << j << std::endl;
                    std::cout << "Dual Value: " << m_dual[clique_index][i][j] << std::endl;
                    std::cout << "Energy Bound: " << energyBound << std::endl;
                    return false;
                }
            }
        }
        clique_index++;
    }
    return true;
}

bool SubmodularPrimalDual1::CheckActiveInvariant() {
    size_t clique_index = 0;
    for (const CliquePtr& cp : m_cliques) {
        const Clique& c = *cp;
        std::vector<NodeId> nodes = c.Nodes();
        const size_t k = nodes.size();
        for (size_t i = 0; i < k; ++i) {
            if (m_dual[clique_index][i][m_labels[nodes[i]]] < -EPS) {
                std::cout << "CliqueId: " << clique_index << std::endl;
                std::cout << "NodeId (w.r.t. Clique): " << i << std::endl;
                std::cout << "Dual Value: " << m_dual[clique_index][i][m_labels[nodes[i]]] << std::endl;
                return false;
            }
        }
        clique_index++;
    }
    return true;
}

bool SubmodularPrimalDual1::CheckZeroSumInvariant() {
    size_t clique_index = 0;
    for (const CliquePtr& cp : m_cliques) {
        const Clique& c = *cp;
        const size_t k = c.Nodes().size();
        for (Label i = 0; i < m_num_labels; ++i) {
            REAL dualSum = 0;
            for (size_t j = 0; j < k; ++j) {
                dualSum += m_dual[clique_index][j][i];
            }
            if (fabs(dualSum) > EPS) {
                std::cout << "CliqueId: " << clique_index << std::endl;
                std::cout << "Label: " << i << std::endl;
                std::cout << "Dual Sum: " << dualSum << std::endl;
                return false;
            }
        }
        clique_index++;
    }
    return true;
}
