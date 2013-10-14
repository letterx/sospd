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
		int k = nodes.size();
		labelBuf.clear();
		for (int i = 0; i < k; ++i) {
			labelBuf.push_back(m_labels[nodes[i]]);
		}
		REAL energy = c.Energy(labelBuf);
        m_dual.push_back(Dual());
		Dual& newDual = m_dual.back();
		newDual.clear();
		for (int i = 0; i < k; ++i) {
			newDual.push_back(std::vector<REAL>(m_num_labels, 0));
		}
        ASSERT(energy >= 0);
        REAL avg = energy / k;
        int remainder = energy % k;
        for (int i = 0; i < k; ++i) {
            newDual[i][m_labels[nodes[i]]] = avg;
            if (i < remainder) // Have to distribute remainder to maintain average
                newDual[i][m_labels[nodes[i]]] += 1;
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
    for (const CliquePtr& cp : m_cliques) {
        const Clique& c = *cp;
        const size_t k = c.Nodes().size();
        ASSERT(k < 32);
        label_buf.resize(k);
        for (size_t i = 0; i < k; ++i) {
            label_buf[i] = m_labels[c.Nodes()[i]];
        }
        REAL energy = c.Energy(label_buf);
        REAL lambdaA = 0;
        REAL lambdaB = 0;
        for (size_t i = 0; i < k; ++i) {
            lambdaA += m_dual[clique_index][i][label_buf[i]];
        }
        std::vector<REAL> psi;
        REAL oldG = energy - lambdaA;
        //This ordering here is important!
        for (int i = k - 1; i >= 0; --i){
            lambdaA -= m_dual[clique_index][i][label_buf[i]];
            lambdaB += m_dual[clique_index][i][alpha];
            label_buf[i] = alpha;
            energy = c.Energy(label_buf);
            REAL newG = energy - lambdaA - lambdaB;
            psi.push_back(oldG - newG);
            oldG = newG;
        }
        for (size_t i = 0; i < k; ++i) {
            m_dual[clique_index][i][alpha] -= psi[k - i - 1];
        }
        ++clique_index;
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

void SubmodularPrimalDual2::SetupGraph(SubmodularIBFS& crf) {
    typedef int32_t Assgn;
    const size_t n = m_labels.size();
    crf.AddNode(n);

    size_t clique_index = 0;
    for (const CliquePtr& cp : m_cliques) {
        const Clique& c = *cp;
        const size_t k = c.Size();
        ASSERT(k < 32);
        const Assgn max_assgn = 1 << k;
        std::vector<REAL> energy_table(max_assgn, 0);
        crf.AddClique(c.Nodes(), energy_table);
        ++clique_index;
    }

    crf.GraphInit();
}

void SubmodularPrimalDual2::SetupAlphaEnergy(Label alpha, SubmodularIBFS& crf) {
    typedef int32_t Assgn;
    const size_t n = m_labels.size();
    crf.ClearUnaries();
    crf.AddConstantTerm(-crf.GetConstantTerm());
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
    SubmodularIBFS::CliqueVec& ibfs_cliques = crf.GetCliques();
    std::vector<Label> label_buf;
    for (const CliquePtr& cp : m_cliques) {
        const Clique& c = *cp;
        auto& ibfs_c = *ibfs_cliques[clique_index];
        const size_t k = c.Size();
        ASSERT(k < 32);
        ASSERT(k == ibfs_c.Size());
        label_buf.resize(k);
        auto& lambda_C = m_dual[clique_index];

        const Assgn max_assgn = 1 << k;
        std::vector<REAL>& energy_table = ibfs_c.EnergyTable();
        for (Assgn a = 0; a < max_assgn; ++a) {
            REAL lambda = 0;
            for (size_t i_idx = 0; i_idx < k; ++i_idx) {
                if (a & (1 << i_idx)) {
                    label_buf[i_idx] = alpha;
                    lambda += lambda_C[i_idx][alpha];
                }
                else {
                    Label x = m_labels[c.Nodes()[i_idx]];
                    label_buf[i_idx] = x;
                    lambda += lambda_C[i_idx][x];
                }
            }
            energy_table[a] = c.Energy(label_buf) - lambda;
        }
        ++clique_index;
    }
}

bool SubmodularPrimalDual2::UpdatePrimalDual(Label alpha, SubmodularIBFS& crf) {
    bool ret = false;
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
        int k = nodes.size();
        labelBuf.clear();
		for (int i = 0; i < k; ++i) {
			labelBuf.push_back(m_labels[nodes[i]]);
		}
		REAL energy = c.Energy(labelBuf);
        REAL avg = energy / k;
        int remainder = energy % k;
		for (int i = 0; i < k; ++i) {
		    m_dual[clique_index][i][labelBuf[i]] = avg;
            if (i < remainder)
                m_dual[clique_index][i][labelBuf[i]] += 1;
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
	#endif
	m_num_cliques = m_cliques.size();
	ComputeRho();
    SubmodularIBFS crf;
    SetupGraph(crf);
	InitialLabeling();
	InitialDual();
	InitialNodeCliqueList();
	#ifdef PROGRESS_DISPLAY
		size_t num_round = 0;
		REAL energy = ComputeEnergy();
		std::cout << "Iteration " << num_round << ": " << energy << std::endl;
	#endif
	#ifdef CHECK_INVARIANTS
        ASSERT(CheckLabelInvariant());
        ASSERT(CheckDualBoundInvariant());
        ASSERT(CheckActiveInvariant());
	#endif
	bool labelChanged = true;
	while (labelChanged){
		labelChanged = false;
		for (size_t alpha = 0; alpha < m_num_labels; ++alpha){
			PreEditDual(alpha);
			#ifdef CHECK_INVARIANTS
                ASSERT(CheckLabelInvariant());
                ASSERT(CheckDualBoundInvariant());
                ASSERT(CheckActiveInvariant());
	        #endif
			if (UpdatePrimalDual(alpha, crf)) labelChanged = true;
			PostEditDual();
			#ifdef CHECK_INVARIANTS
                ASSERT(CheckLabelInvariant());
                ASSERT(CheckDualBoundInvariant());
                ASSERT(CheckActiveInvariant());
	        #endif
		}
		#ifdef PROGRESS_DISPLAY
			energy = ComputeEnergy();
			num_round++;
			std::cout << "Iteration " << num_round << ": " << energy << std::endl;
		#endif
	}
	#ifdef CHECK_INVARIANTS
	    ASSERT(CheckHeightInvariant());
	#endif
	DualFit();
    #ifdef PROGRESS_DISPLAY
	    std::cout << ")" << std::endl;
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

void SubmodularPrimalDual2::ComputeRho() {
    m_rho = 1;
    for (const CliquePtr& cp : m_cliques) {
        Clique& c = *cp;
        m_rho = std::max(m_rho, c.Rho());
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
            if (hx > halpha) {
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
        if (abs(sum - energy)) {
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
                if (m_dual[clique_index][i][j] > energyBound) {
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
            if (m_dual[clique_index][i][m_labels[nodes[i]]] < 0) {
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

/*
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
            if (abs(dualSum) > 0) {
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
*/
