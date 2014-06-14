#include "sospd.hpp"
#include "multilabel-energy.hpp"

SoSPD::SoSPD(const MultilabelEnergy* energy)
    : m_energy(energy),
    m_ibfs(),
    m_num_labels(energy->numLabels()),
    m_labels(energy->numVars(), 0),
    m_fusion_labels(energy->numVars(), 0),
    m_dual(),
    m_heights(),
    m_expansion_submodular(false),
    m_lower_bound(true),
    m_iter(0),
    m_pc([&](int, const std::vector<Label>&, std::vector<Label>&) { HeightAlphaProposal(); })
{ }

int SoSPD::GetLabel(VarId i) const {
    return m_labels[i];
}

void SoSPD::InitialLabeling() {
    const VarId n = m_energy->numVars();
    for (VarId i = 0; i < n; ++i) {
        REAL best_cost = std::numeric_limits<REAL>::max();
        for (size_t l = 0; l < m_num_labels; ++l) {
            if (m_energy->unary(i, l) < best_cost) {
                best_cost = m_energy->unary(i, l);
                m_labels[i] = l;
            }
        }
    }
}

void SoSPD::InitialDual() {
    // Initialize heights
    m_heights = std::vector<REAL>(m_energy->numVars()*m_num_labels, 0);
    for (VarId i = 0; i < m_energy->numVars(); ++i)
        for (Label l = 0; l < m_num_labels; ++l)
            Height(i, l) = m_energy->unary(i, l);

    m_dual.clear();
    Label labelBuf[32];
    for (const CliquePtr& cp : m_energy->cliques()) {
        const Clique& c = *cp;
		const VarId* nodes = c.nodes();
		int k = c.size();
        ASSERT(k < 32);
		for (int i = 0; i < k; ++i) {
            labelBuf[i] = m_labels[nodes[i]];
		}
		REAL energy = c.energy(labelBuf);
        m_dual.emplace_back(k*m_num_labels, 0);
		LambdaAlpha& lambda_a = m_dual.back();
        
        ASSERT(energy >= 0);
        REAL avg = energy / k;
        int remainder = energy % k;
        for (int i = 0; i < k; ++i) {
            Label l = m_labels[nodes[i]];
            REAL& lambda_ail = dualVariable(lambda_a, i, l);
            lambda_ail = avg;
            if (i < remainder) // Have to distribute remainder to maintain average
                lambda_ail += 1;
            Height(nodes[i], l) += lambda_ail;
        }
    }
}

void SoSPD::InitialNodeCliqueList() {
    size_t n = m_labels.size();
    m_node_clique_list.clear();
    m_node_clique_list.resize(n);

    int clique_index = 0;
    for (const CliquePtr& cp : m_energy->cliques()) {
        const Clique& c = *cp;
        const VarId* nodes = c.nodes();
        const size_t k = c.size();
        for (size_t i = 0; i < k; ++i) {
            m_node_clique_list[nodes[i]].push_back(std::make_pair(clique_index, i));
        }
        ++clique_index;
    }
}

void SoSPD::PreEditDual(SubmodularIBFS& crf) {
    // Allocate all the buffers we need in one place, resize as necessary
    Label label_buf[32];
    std::vector<Label> current_labels;
    std::vector<Label> fusion_labels;
    std::vector<REAL> psi;
    std::vector<REAL> current_lambda;
    std::vector<REAL> fusion_lambda;

    SubmodularIBFS::CliqueVec& ibfs_cliques = crf.GetCliques();
    int clique_index = 0;
    for (const CliquePtr& cp : m_energy->cliques()) {
        const Clique& c = *cp;
        const size_t k = c.size();
        ASSERT(k < 32);

        auto& lambda_a = lambdaAlpha(clique_index);

        auto& ibfs_c = ibfs_cliques[clique_index];
        ASSERT(k == ibfs_c.Size());
        std::vector<REAL>& energy_table = ibfs_c.EnergyTable();
        Assgn max_assgn = 1 << k;
        ASSERT(energy_table.size() == max_assgn);

        psi.resize(k);
        current_labels.resize(k);
        fusion_labels.resize(k);
        current_lambda.resize(k);
        fusion_lambda.resize(k);
        for (size_t i = 0; i < k; ++i) {
            current_labels[i] = m_labels[c.nodes()[i]];
            fusion_labels[i] = m_fusion_labels[c.nodes()[i]];
            current_lambda[i] = dualVariable(lambda_a, i, current_labels[i]);
            fusion_lambda[i] = dualVariable(lambda_a, i, fusion_labels[i]);
        }
        
        // Compute costs of all fusion assignments
        {
            Assgn last_gray = 0;
            for (size_t i_idx = 0; i_idx < k; ++i_idx)
                label_buf[i_idx] = current_labels[i_idx];
            energy_table[0] = c.energy(label_buf);
            for (Assgn a = 1; a < max_assgn; ++a) {
                Assgn gray = a ^ (a >> 1);
                Assgn diff = gray ^ last_gray;
                int changed_idx = __builtin_ctz(diff);
                if (diff & gray)
                    label_buf[changed_idx] = fusion_labels[changed_idx];
                else
                    label_buf[changed_idx] = current_labels[changed_idx];
                last_gray = gray;
                energy_table[gray] = c.energy(label_buf);
            }
        }

        if (!m_expansion_submodular) {
            // Find g with g(S) >= f(S) and g submodular. Also want to make sure
            // that g(S | T) == g(S) where T is the set of nodes with 
            // current[i] == fusion[i]
            if (m_lower_bound)
                SubmodularLowerBound(k, energy_table);
            else
                SubmodularUpperBound(k, energy_table);
            Assgn fusion_equals_current = 0;
            for (size_t i = 0; i < k; ++i) {
                if (current_labels[i] == fusion_labels[i])
                    fusion_equals_current |= (1 << i);
            }
            ZeroMarginalSet(k, energy_table, fusion_equals_current);
            //ASSERT(CheckUpperBoundInvariants(k, energy_table, upper_bound));
        }

        // Compute the residual function 
        // g(S) - lambda_fusion(S) - lambda_current(C\S)
        SubtractLinear(k, energy_table, fusion_lambda, current_lambda);
        ASSERT(energy_table[0] == 0); // Check tightness of current labeling

        // Modify g, find psi so that g(S) + psi(S) >= 0
        Normalize(k, energy_table, psi);

        // Update lambda_fusion[i] so that 
        // g(S) - lambda_fusion(S) - lambda_current(C\S) >= 0
        for (size_t i = 0; i < k; ++i) {
            dualVariable(lambda_a, i, fusion_labels[i]) -= psi[i];
            Height(c.nodes()[i], fusion_labels[i]) -= psi[i];
        }

        ++clique_index;
    }
}

REAL SoSPD::ComputeHeight(VarId i, Label x) {
    REAL ret = m_energy->unary(i, x);
    for (const auto& p : m_node_clique_list[i]) {
        ret += dualVariable(p.first, p.second, x);
    }
    return ret;
}

REAL SoSPD::ComputeHeightDiff(VarId i, Label l1, Label l2) const {
    REAL ret = m_energy->unary(i, l1) - m_energy->unary(i, l2);
    for (const auto& p : m_node_clique_list[i]) {
        ret += dualVariable(p.first, p.second, l1) 
            - dualVariable(p.first, p.second, l2);
    }
    return ret;
}

void SoSPD::SetupGraph(SubmodularIBFS& crf) {
    typedef int32_t Assgn;
    const size_t n = m_labels.size();
    crf.AddNode(n);

    size_t clique_index = 0;
    for (const CliquePtr& cp : m_energy->cliques()) {
        const Clique& c = *cp;
        const size_t k = c.size();
        ASSERT(k < 32);
        const Assgn max_assgn = 1 << k;
        std::vector<SubmodularIBFS::NodeId> nodes(c.nodes(), c.nodes() + c.size());
        crf.AddClique(nodes, std::vector<REAL>(max_assgn, 0), false);
        ++clique_index;
    }

    crf.GraphInit();
}

void SoSPD::SetupAlphaEnergy(SubmodularIBFS& crf) {
    typedef int32_t Assgn;
    const size_t n = m_labels.size();
    crf.ClearUnaries();
    crf.AddConstantTerm(-crf.GetConstantTerm());
    for (size_t i = 0; i < n; ++i) {
        REAL height_diff = ComputeHeightDiff(i, m_labels[i], m_fusion_labels[i]);
        if (height_diff > 0) {
            crf.AddUnaryTerm(i, height_diff, 0);
        }
        else {
            crf.AddUnaryTerm(i, 0, -height_diff);
        }
    }
}

bool SoSPD::UpdatePrimalDual(SubmodularIBFS& crf) {
    bool ret = false;
    SetupAlphaEnergy(crf);
    crf.Solve();
    VarId n = m_labels.size();
    for (VarId i = 0; i < n; ++i) {
        int crf_label = crf.GetLabel(i);
        if (crf_label == 1) {
            Label alpha = m_fusion_labels[i];
            if (m_labels[i] != alpha) ret = true;
            m_labels[i] = alpha;
        }
    }
    const SubmodularIBFS::CliqueVec& clique = crf.GetCliques();
    size_t i = 0;
    for (const CliquePtr& cp : m_energy->cliques()) {
        const Clique& c = *cp;
        auto& ibfs_c = clique[i];
        const std::vector<REAL>& phiCi = ibfs_c.AlphaCi();
        for (size_t j = 0; j < phiCi.size(); ++j) {
            dualVariable(i, j, m_fusion_labels[c.nodes()[j]]) += phiCi[j];
            Height(c.nodes()[j], m_fusion_labels[c.nodes()[j]]) += phiCi[j];
        }
        ++i;
    }
    return ret;
}

void SoSPD::PostEditDual() {
    Label labelBuf[32];
    int clique_index = 0;
    for (const CliquePtr& cp : m_energy->cliques()) {
        const Clique& c = *cp;
        const VarId* nodes = c.nodes();
        int k = c.size();
        ASSERT(k < 32);
		for (int i = 0; i < k; ++i) {
            labelBuf[i] = m_labels[nodes[i]];
		}
		REAL energy = c.energy(labelBuf);
        REAL avg = energy / k;
        int remainder = energy % k;
		for (int i = 0; i < k; ++i) {
            auto& lambda_ail = dualVariable(clique_index,  i, labelBuf[i]);
            Height(nodes[i], labelBuf[i]) -= lambda_ail;
		    lambda_ail = avg;
            if (i < remainder)
                lambda_ail += 1;
            Height(nodes[i], labelBuf[i]) += lambda_ail;
		}
		++clique_index;
    }
}

void SoSPD::DualFit() {
    // FIXME: This is the only function that doesn't work with integer division.
    // It's also not really used for anything at the moment
    /*
	for (size_t i = 0; i < m_dual.size(); ++i)
		for (size_t j = 0; j < m_dual[i].size(); ++j)
			for (size_t k = 0; k < m_dual[i][j].size(); ++k)
				m_dual[i][j][k] /= (m_mu * m_rho);
                */
    ASSERT(false /* unimplemented */);
}

bool SoSPD::InitialFusionLabeling() {
    m_pc(m_iter, m_labels, m_fusion_labels);
    for (size_t i = 0; i < m_labels.size(); ++i) {
        if (m_labels[i] != m_fusion_labels[i])
            return true;
    }
    return false;
}

void SoSPD::HeightAlphaProposal() {
    const size_t n = m_labels.size();
    REAL max_s_capacity = 0;
    Label alpha = 0;
    for (Label l = 0; l < m_num_labels; ++l) {
        REAL s_capacity = 0;
        for (size_t i = 0; i < n; ++i) {
            REAL diff = Height(i, m_labels[i]) - Height(i, l);
            if (diff > 0)
                s_capacity += diff;
        }
        if (s_capacity > max_s_capacity) {
            max_s_capacity = s_capacity;
            alpha = l;
        }
    }
    for (size_t i = 0; i < n; ++i)
        m_fusion_labels[i] = alpha;
}

void SoSPD::AlphaProposal() {
    Label alpha = m_iter % m_num_labels;
    const size_t n = m_labels.size();
    for (size_t i = 0; i < n; ++i)
        m_fusion_labels[i] = alpha;
}


void SoSPD::Solve(int niters) {
    if (m_iter == 0) {
        SetupGraph(m_ibfs);
        InitialLabeling();
        InitialDual();
        InitialNodeCliqueList();
    }
	#ifdef PROGRESS_DISPLAY
		REAL energy = m_energy->ComputeEnergy(m_labels);
		std::cout << "Iteration " << m_iter << ": " << energy << std::endl;
	#endif
	#ifdef CHECK_INVARIANTS
        ASSERT(CheckLabelInvariant());
        ASSERT(CheckActiveInvariant());
	#endif
	bool labelChanged = true;
    int this_iter = 0;
	while (labelChanged && this_iter < niters){
        labelChanged = InitialFusionLabeling();
        if (!labelChanged) break;
	    PreEditDual(m_ibfs);
		#ifdef CHECK_INVARIANTS
            ASSERT(CheckLabelInvariant());
            ASSERT(CheckActiveInvariant());
	    #endif
        UpdatePrimalDual(m_ibfs);
		PostEditDual();
		#ifdef CHECK_INVARIANTS
            ASSERT(CheckLabelInvariant());
            ASSERT(CheckActiveInvariant());
	    #endif
        this_iter++;
        m_iter++;
		#ifdef PROGRESS_DISPLAY
			energy = m_energy->ComputeEnergy(m_labels);
			std::cout << "Iteration " << m_iter << ": " << energy << std::endl;
		#endif
	}
	#ifdef CHECK_INVARIANTS
    if (!labelChanged)
	    ASSERT(CheckHeightInvariant());
	#endif
    //LowerBound();
}

bool SoSPD::CheckHeightInvariant() {
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

bool SoSPD::CheckLabelInvariant() {
    size_t clique_index = 0;
    Label labelBuf[32];
    for (const CliquePtr& cp : m_energy->cliques()) {
        const Clique& c = *cp;
        const VarId* nodes = c.nodes();
        const size_t k = c.size();
        ASSERT(k < 32);
        for (size_t i = 0; i < k; ++i) {
            labelBuf[i] = m_labels[nodes[i]];
        }
        REAL energy = c.energy(labelBuf);
        REAL sum = 0;
        for (size_t i = 0; i < k; ++i) {
            sum += dualVariable(clique_index, i, labelBuf[i]);
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

bool SoSPD::CheckActiveInvariant() {
    size_t clique_index = 0;
    for (const CliquePtr& cp : m_energy->cliques()) {
        const Clique& c = *cp;
        const VarId* nodes = c.nodes();
        const size_t k = c.size();
        for (size_t i = 0; i < k; ++i) {
            
            if (dualVariable(clique_index, i, m_labels[nodes[i]]) < 0) {
                std::cout << "CliqueId: " << clique_index << std::endl;
                std::cout << "VarId (w.r.t. Clique): " << i << std::endl;
                std::cout << "Dual Value: " 
                    << dualVariable(clique_index, i, m_labels[nodes[i]]) 
                    << std::endl;
                return false;
            }
        }
        clique_index++;
    }
    return true;
}

REAL SoSPD::dualVariable(int alpha, VarId i, Label l) const {
    return m_dual[alpha][i*m_num_labels+l];
}

REAL& SoSPD::dualVariable(int alpha, VarId i, Label l) {
    return m_dual[alpha][i*m_num_labels+l];
}

REAL SoSPD::dualVariable(const LambdaAlpha& lambdaAlpha, 
        VarId i, Label l) const {
    return lambdaAlpha[i*m_num_labels+l];
}

REAL& SoSPD::dualVariable(LambdaAlpha& lambdaAlpha, 
        VarId i, Label l) {
    return lambdaAlpha[i*m_num_labels+l];
}

SoSPD::LambdaAlpha& SoSPD::lambdaAlpha(int alpha) {
    return m_dual[alpha];
}

const SoSPD::LambdaAlpha& SoSPD::lambdaAlpha(int alpha) const {
    return m_dual[alpha];
}

/*
bool SoSPD::CheckZeroSumInvariant() {
    size_t clique_index = 0;
    for (const CliquePtr& cp : m_energy->cliques()) {
        const Clique& c = *cp;
        const size_t k = c.nodes().size();
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

double SoSPD::LowerBound() {
    std::cout << "Computing Lower Bound\n";
    double max_ratio = 0;
    int clique_index = 0;
    for (const CliquePtr& cp : m_energy->cliques()) {
        const Clique& c = *cp;
        const size_t k = c.size();
        ASSERT(k == 3); // Lower bound doesn't work for larger numbers
        Label buf[3];
        for (buf[0] = 0; buf[0] < m_num_labels; ++buf[0]) {
            for (buf[1] = 0; buf[1] < m_num_labels; ++buf[1]) {
                for (buf[2] = 0; buf[2] < m_num_labels; ++buf[2]) {
                    REAL energy = c.energy(buf);
                    REAL dualSum = dualVariable(clique_index, 0, buf[0])
                        + dualVariable(clique_index, 1, buf[1])
                        + dualVariable(clique_index, 2, buf[2]);
                    if (energy == 0) {
                        for (int i = 0; i < 3; ++i) {
                            if (buf[i] != m_labels[c.nodes()[i]]) {
                                dualVariable(clique_index, i, buf[i]) -= dualSum - energy;
                                Height(c.nodes()[i], buf[i]) -= dualSum - energy;
                                dualSum = energy;
                                break;
                            }
                        }
                        ASSERT(dualSum == energy);
                    } else {
                        max_ratio = std::max(max_ratio, double(dualSum)/double(energy));
                    }
                }
            }
        }
        clique_index++;
    }
    REAL dual_objective = 0;
    for (VarId i = 0; i < m_energy->numVars(); ++i) {
        REAL min_height = std::numeric_limits<REAL>::max();
        for (Label l = 0; l < m_num_labels; ++l)
            min_height = std::min(min_height, Height(i, l));
        dual_objective += min_height;
    }
    std::cout << "Max Ratio: " << max_ratio << "\n";
    std::cout << "Dual objective: " << dual_objective << "\n";
    return dual_objective / max_ratio;
}
