#include "spd2.hpp"
#include "clique.hpp"

SubmodularPrimalDual2::SubmodularPrimalDual2(Label max_label)
    : m_num_labels(max_label),
    m_constant_term(0),
    m_cliques(),
    m_unary_cost(),
    m_labels()
{ }

SubmodularPrimalDual2::NodeId SubmodularPrimalDual2::AddNode(int n) {
    NodeId ret = m_labels.size();
    for (int i = 0; i < n; ++i) {
        m_labels.push_back(0);
        UnaryCost uc(m_num_labels, 0);
        m_unary_cost.push_back(uc);
    }
    return ret;
}

int SubmodularPrimalDual2::GetLabel(NodeId i) const {
    return m_labels[i];
}

void SubmodularPrimalDual2::AddConstantTerm(REAL c) {
    m_constant_term += c;
}

void SubmodularPrimalDual2::AddUnaryTerm(NodeId i, const std::vector<REAL>& coeffs) {
    ASSERT(coeffs.size() == m_num_labels);
    for (size_t j = 0; j < m_num_labels; ++j) {
        m_unary_cost[i][j] += coeffs[j];
    }
}

void SubmodularPrimalDual2::AddClique(const CliquePtr& cp) {
    m_cliques.push_back(cp);
}

void SubmodularPrimalDual2::InitialLabeling() {
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

void SubmodularPrimalDual2::InitialDual() {
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
		REAL energy = c.Energy(labelBuf);
        m_dual.push_back(Dual());
		Dual& newDual = m_dual.back();
		newDual.clear();
		for (size_t i = 0; i < k; ++i) {
			newDual.push_back(std::vector<REAL>(m_num_labels, 0));
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
    }
}

void SubmodularPrimalDual2::InitialNodeCliqueList() {
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

void SubmodularPrimalDual2::PreEditDual(Label alpha) {
    std::vector<Label> label_buf;
    int clique_index = 0;
    //bool t = true;
    for (const CliquePtr& cp : m_cliques) {
        const Clique& c = *cp;
        const size_t k = c.Nodes().size();
        ASSERT(k < 32);
        label_buf.clear();
        for (size_t i = 0; i < k; ++i) {
            label_buf.push_back(m_labels[c.Nodes()[i]]);
        }
        REAL energy = c.Energy(label_buf);
        REAL lambdaA = 0;
        REAL lambdaB = 0;
        for (size_t i = 0; i < k; ++i) {
            lambdaA += m_dual[clique_index][i][label_buf[i]];
        }
        std::vector<REAL> psi;
        REAL oldG = energy - lambdaA;
        /*if ((k == 4) && t) {
            for (size_t i = 0; i < k; ++i)
                std::cout << label_buf[i] << " ";
            std::cout << std::endl;
            for (size_t i = 0; i < k; ++i)
                std::cout << m_dual[clique_index][i][label_buf[i]] << " ";
            std::cout << std::endl;
            for (size_t i = 0; i < k; ++i)
                std::cout << m_dual[clique_index][i][alpha] << " ";
            std::cout << std::endl;
        }*/
        //if ((k == 4) && t) std::cout << oldG << " ";
        //This ordering here is important!
        for (int i = k - 1; i >= 0; --i){
            lambdaA -= m_dual[clique_index][i][label_buf[i]];
            lambdaB += m_dual[clique_index][i][alpha];
            label_buf[i] = alpha;
            energy = c.Energy(label_buf);
            REAL newG = energy - lambdaA - lambdaB;
            psi.push_back(oldG - newG);
            oldG = newG;
            //if ((k == 4) && t) std::cout << oldG << " ";
        }
        //if ((k == 4) && t) std::cout << std::endl;
        for (size_t i = 0; i < k; ++i) {
            m_dual[clique_index][i][alpha] -= psi[k - i - 1];
        }
        /*if ((k == 4) && t) {
            for (size_t i = 0; i < k; ++i)
                std::cout << psi[i] << " ";
            std::cout << std::endl;
            for (size_t i = 0; i < k; ++i)
                std::cout << m_dual[clique_index][i][alpha] << " ";
            std::cout << std::endl;
        }*/
        ++clique_index;
        //if (k == 4) t = false;
    }
}

REAL SubmodularPrimalDual2::ComputeHeight(NodeId i, Label x) {
    REAL ret = m_unary_cost[i][x];
    for (size_t j = 0; j < m_node_clique_list[i].size(); ++j) {
        size_t cliqueId = m_node_clique_list[i][j].first;
        size_t nodeId = m_node_clique_list[i][j].second;
        ret += m_dual[cliqueId][nodeId][x];
    }
    return ret;
}

void SubmodularPrimalDual2::SetupAlphaEnergy(Label alpha, SubmodularIBFS& crf) {
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
    //bool t = true;
    for (const CliquePtr& cp : m_cliques) {
        const Clique& c = *cp;
        const size_t k = c.Size();
        ASSERT(k < 32);
        const Assgn max_assgn = 1 << k;
        std::vector<REAL> energy_table;
        std::vector<Label> label_buf;
        /*if ((k == 4) && t) {
            for (size_t i = 0; i < k; ++i)
                std::cout << m_dual[clique_index][i][alpha] << " ";
            std::cout << std::endl;
        }*/
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
        /*if ((k == 4) && t){
            for (int i = 0; i < max_assgn; ++i) std::cout << energy_table[i] << " ";
            std::cout << std::endl;
        }*/
        crf.AddClique(c.Nodes(), energy_table);
        ++clique_index;
        //if (k == 4) t = false;
    }
}

bool SubmodularPrimalDual2::UpdatePrimalDual(Label alpha) {
    bool ret = false;
    SubmodularIBFS crf;
    SetupAlphaEnergy(alpha, crf);
    crf.Solve();
    NodeId n = m_labels.size();
    for (NodeId i = 0; i < n; ++i) {
        int crf_label = crf.GetLabel(i);
        if (crf_label == 1) {
            if (m_labels[i] != alpha) ret = true;
            m_labels[i] = alpha;
        }
    }
    SubmodularIBFS::CliqueVec clique = crf.GetCliques();
    for (size_t i = 0; i < m_num_cliques; ++i) {
        SubmodularIBFS::CliquePtr c = clique[i];
        std::vector<REAL> phiCi = c->AlphaCi();
        for (size_t j = 0; j < phiCi.size(); ++j) {
            m_dual[i][j][alpha] += phiCi[j];
        }
    }
    return ret;
}

void SubmodularPrimalDual2::PostEditDual() {
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
		REAL energy = c.Energy(labelBuf);
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

void SubmodularPrimalDual2::DualFit() {
    // FIXME: This is the only function that doesn't work with integer division.
    // It's also not really used for anything at the moment
    /*
	for (size_t i = 0; i < m_dual.size(); ++i)
		for (size_t j = 0; j < m_dual[i].size(); ++j)
			for (size_t k = 0; k < m_dual[i][j].size(); ++k)
				m_dual[i][j][k] /= (m_mu * m_rho);
                */
}

void SubmodularPrimalDual2::Solve() {
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
	    //std::cout << m_dual[728][1][2] << std::endl;
        ASSERT(CheckLabelInvariant());
        ASSERT(CheckDualBoundInvariant());
        ASSERT(CheckActiveInvariant());
        ASSERT(CheckZeroSumInvariant());
	#endif
	bool labelChanged = true;
	while (labelChanged){
		labelChanged = false;
		for (size_t alpha = 0; alpha < m_num_labels; ++alpha){
			PreEditDual(alpha);
			#ifdef DEBUG
			    std::cout << "Pre-edit Done!" << std::endl;
                ASSERT(CheckLabelInvariant());
                ASSERT(CheckDualBoundInvariant());
                ASSERT(CheckActiveInvariant());
                ASSERT(CheckZeroSumInvariant());
	        #endif
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
	DualFit();
    #ifdef PROGRESS_DISPLAY
	    std::cout << ")" << std::endl;
	    std::cout.flush();
    #endif
}

REAL SubmodularPrimalDual2::ComputeEnergy() const {
    return ComputeEnergy(m_labels);
}

REAL SubmodularPrimalDual2::ComputeEnergy(const std::vector<Label>& labels) const {
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

void SubmodularPrimalDual2::SetMu(double mu) {
}

double SubmodularPrimalDual2::GetMu() {
    return 1;
}
        
void SubmodularPrimalDual2::ComputeRho() {
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

double SubmodularPrimalDual2::GetRho() {
    return m_rho;
}

bool SubmodularPrimalDual2::CheckHeightInvariant() {
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

bool SubmodularPrimalDual2::CheckLabelInvariant() {
    size_t clique_index = 0;
    for (const CliquePtr& cp : m_cliques) {
        const Clique& c = *cp;
        std::vector<NodeId> nodes = c.Nodes();
        const size_t k = nodes.size();
        std::vector<Label> labelBuf;
        for (size_t i = 0; i < k; ++i) {
            labelBuf.push_back(m_labels[nodes[i]]);
        }
        REAL energy = c.Energy(labelBuf);
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

bool SubmodularPrimalDual2::CheckDualBoundInvariant() {
    size_t clique_index = 0;
    for (const CliquePtr& cp : m_cliques) {
        const Clique& c = *cp;
        REAL energyBound = c.m_f_max;
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

bool SubmodularPrimalDual2::CheckActiveInvariant() {
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

bool SubmodularPrimalDual2::CheckZeroSumInvariant() {
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
