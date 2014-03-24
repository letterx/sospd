#include <boost/test/unit_test.hpp>
#include <iostream>
#include "submodular-flow.hpp"
#include "higher-order-energy.hpp"
#include "gen-random.hpp"
#include "qpbo.hpp"

typedef SubmodularFlow::NodeId NodeId;
typedef SubmodularFlow::CliqueId CliqueId;

/* Sets up a submodular flow problem with a single clique. The clique has 4
 * nodes, and is equal to -1 when all 4 nodes are set to 1, 0 otherwise.
 */
void SetupMinimalFlow(SubmodularFlow& sf) {

    sf.AddNode(4);

    std::vector<NodeId> nodes = {0, 1, 2, 3};
    const size_t numAssignments = 1 << 4;
    std::vector<REAL> energyTable(numAssignments, 0);
    energyTable[0xf] = -1;
    sf.AddClique(nodes, energyTable);
}

BOOST_AUTO_TEST_SUITE(setupTests)

BOOST_AUTO_TEST_CASE(defaultConstructor) {
    SubmodularFlow sf;

    BOOST_CHECK_EQUAL(sf.GetConstantTerm(), 0);
    BOOST_CHECK_EQUAL(sf.GetNumNodes(), 0);
    BOOST_CHECK_EQUAL(sf.GetC_si().size(), 0);
    BOOST_CHECK_EQUAL(sf.GetC_it().size(), 0);
    BOOST_CHECK_EQUAL(sf.GetPhi_si().size(), 0);
    BOOST_CHECK_EQUAL(sf.GetPhi_it().size(), 0);
    BOOST_CHECK_EQUAL(sf.GetLabels().size(), 0);
    BOOST_CHECK_EQUAL(sf.GetNumCliques(), 0);
    BOOST_CHECK_EQUAL(sf.GetCliques().size(), 0);
    BOOST_CHECK_EQUAL(sf.GetNeighbors().size(), 0);
    BOOST_CHECK_EQUAL(sf.ComputeEnergy(), 0);
}

BOOST_AUTO_TEST_CASE(minimalFlowSetup) {
    SubmodularFlow sf;
    SetupMinimalFlow(sf);

    BOOST_CHECK_EQUAL(sf.GetConstantTerm(), -1);
    BOOST_CHECK_EQUAL(sf.GetC_si().size(), 4);
    BOOST_CHECK_EQUAL(sf.GetC_it().size(), 4);
    BOOST_CHECK_EQUAL(sf.GetPhi_si().size(), 4);
    BOOST_CHECK_EQUAL(sf.GetPhi_it().size(), 4);
    BOOST_CHECK_EQUAL(sf.GetLabels().size(), 4);
    BOOST_CHECK_EQUAL(sf.GetNumCliques(), 1);
    BOOST_CHECK_EQUAL(sf.GetCliques().size(), 1);
    BOOST_CHECK_EQUAL(sf.GetNeighbors().size(), 4);
    BOOST_CHECK_EQUAL(sf.ComputeEnergy(), 0);

    std::vector<int>& labels = sf.GetLabels();
    const uint32_t max_assgn = 1 << 4;
    for (uint32_t assgn = 0; assgn < max_assgn; ++assgn) {
        for (NodeId i = 0; i < 4; ++i) {
            if (assgn & (1 << i))
                labels[i] = 1;
            else
                labels[i] = 0;
        }
        if (assgn == 0xf)
            BOOST_CHECK_EQUAL(sf.ComputeEnergy(), -1);
        else
            BOOST_CHECK_EQUAL(sf.ComputeEnergy(), 0);
    }
}

BOOST_AUTO_TEST_CASE(randomFlowSetup) {
    SubmodularFlow sf;
    const size_t n = 100;
    const size_t k = 4;
    const size_t m = 100;
    const REAL clique_range = 100;
    const REAL unary_mean = 800;
    const REAL unary_var = 1600;
    const unsigned int seed = 0;

    GenRandom(sf, n, k, m, clique_range, unary_mean, unary_var, seed);

    BOOST_CHECK_EQUAL(sf.GetC_si().size(), n);
    BOOST_CHECK_EQUAL(sf.GetC_it().size(), n);
    BOOST_CHECK_EQUAL(sf.GetPhi_si().size(), n);
    BOOST_CHECK_EQUAL(sf.GetPhi_it().size(), n);
    BOOST_CHECK_EQUAL(sf.GetLabels().size(), n);
    BOOST_CHECK_EQUAL(sf.GetNumCliques(), m);
    BOOST_CHECK_EQUAL(sf.GetCliques().size(), m);
    BOOST_CHECK_EQUAL(sf.GetNeighbors().size(), n);

}

/* Check that for a clique c, the energy is always >= 0, and is equal to 0 at
 * the all 0 and all 1 labelings.
 */
void CheckNormalized(const SubmodularFlow::Clique& c, std::vector<int>& labels) {
    const size_t n = c.Nodes().size();
    BOOST_REQUIRE_LT(n, 32);
    const uint32_t max_assgn = 1 << n;
    for (uint32_t assgn = 0; assgn < max_assgn; ++assgn) {
        for (size_t i = 0; i < n; ++i) {
            if (assgn & (1 << i))
                labels[c.Nodes()[i]] = 1;
            else
                labels[c.Nodes()[i]] = 0;
        }
        if (assgn == 0)
            BOOST_CHECK_EQUAL(c.ComputeEnergy(labels), 0);
        else if (assgn == max_assgn - 1)
            BOOST_CHECK_EQUAL(c.ComputeEnergy(labels), 0);
        else
            BOOST_CHECK_GE(c.ComputeEnergy(labels), 0);
    }
}

/* Check that all source-sink capacities are >= 0, and that cliques
 * are normalized (i.e., are >= 0 for all labelings)
 */
BOOST_AUTO_TEST_CASE(randomFlowNormalized) {
    SubmodularFlow sf;
    const size_t n = 100;
    const size_t k = 4;
    const size_t m = 100;
    const REAL clique_range = 100;
    const REAL unary_mean = 800;
    const REAL unary_var = 1600;
    const unsigned int seed = 0;

    GenRandom(sf, n, k, m, clique_range, unary_mean, unary_var, seed);

    for (size_t i = 0; i < n; ++i) {
        BOOST_CHECK_GE(sf.GetC_si()[i], 0);
        BOOST_CHECK_GE(sf.GetC_it()[i], 0);
        BOOST_CHECK_EQUAL(sf.GetPhi_si()[i], 0);
        BOOST_CHECK_EQUAL(sf.GetPhi_it()[i], 0);
    }

    for (const SubmodularFlow::CliquePtr& cp : sf.GetCliques()) {
        const SubmodularFlow::Clique& c = *cp;
        BOOST_CHECK_EQUAL(c.Nodes().size(), 4);
        CheckNormalized(c, sf.GetLabels());
    }
}

BOOST_AUTO_TEST_SUITE_END()



BOOST_AUTO_TEST_SUITE(flowInvariants)

void TestNonnegativeCapacities(const SubmodularFlow& sf) {
    const auto& c_si = sf.GetC_si();
    const auto& c_it = sf.GetC_it();
    const auto& phi_si = sf.GetPhi_si();
    const auto& phi_it = sf.GetPhi_it();

    for (NodeId i = 0; i < sf.GetNumNodes(); ++i) {
        BOOST_REQUIRE_GE(phi_si[i], 0);
        BOOST_REQUIRE_GE(phi_it[i], 0);
        BOOST_REQUIRE_GE(c_si[i], phi_si[i]);
        BOOST_REQUIRE_GE(c_it[i], phi_it[i]);
    }
    for (const auto& cp : sf.GetCliques()) {
        const auto& c = *cp;
        size_t i_idx = 0;
        for (NodeId i : c.Nodes()) {
            size_t j_idx = 0;
            for (NodeId j : c.Nodes()) {
                if (i != j)
                    BOOST_REQUIRE_GE(c.ExchangeCapacity(i_idx, j_idx), 0);
                j_idx++;
            }
            i_idx++;
        }
    }
}

void TestExcess(const SubmodularFlow& sf) {
    const auto& excess = sf.GetExcess();
    const auto& dis = sf.GetDis();
    const auto s = sf.GetS();
    const auto& phi_si = sf.GetPhi_si();
    const auto& phi_it = sf.GetPhi_it();
    const auto& cliques = sf.GetCliques();
    const auto& neighbors = sf.GetNeighbors();

    for (NodeId i = 0; i < sf.GetNumNodes(); ++i) {
        REAL expected_excess = phi_si[i] - phi_it[i];
        for (CliqueId cid : neighbors[i]) {
            const auto& c = *cliques[cid];
            const size_t clique_index = std::find(c.Nodes().begin(), c.Nodes().end(), i) - c.Nodes().begin();
            BOOST_REQUIRE_EQUAL(i, c.Nodes()[clique_index]);
            expected_excess -= c.AlphaCi()[clique_index];
        }
        BOOST_REQUIRE_GE(excess[i], 0);
        if (dis[i] < dis[s])
            BOOST_REQUIRE_EQUAL(expected_excess, excess[i]);
        else 
            BOOST_REQUIRE_EQUAL(excess[i], 0);
    }
}

void TestDistance(const SubmodularFlow& sf) {
    const auto& dis = sf.GetDis();
    const auto& cliques = sf.GetCliques();
    const auto& c_si = sf.GetC_si();
    const auto& c_it = sf.GetC_it();
    const auto& phi_si = sf.GetPhi_si();
    const auto& phi_it = sf.GetPhi_it();
    NodeId s = sf.GetS();
    NodeId t = sf.GetT();

    BOOST_REQUIRE_EQUAL(dis[s], sf.GetNumNodes()+2);
    BOOST_REQUIRE_EQUAL(dis[t], 0);
    
    for (NodeId i = 0; i < sf.GetNumNodes(); ++i) {
        BOOST_REQUIRE_GE(dis[i], 0);
        if (c_si[i] - phi_si[i] > 0) BOOST_REQUIRE_LE(dis[s], dis[i]+1);
        if (phi_si[i] > 0) BOOST_REQUIRE_LE(dis[i], dis[s]+1);
        if (c_it[i] - phi_it[i] > 0) BOOST_REQUIRE_LE(dis[i], dis[t]+1);
    }
    for (const auto& cp : cliques) {
        const auto& c = *cp;
        size_t i_idx = 0;
        for (NodeId i : c.Nodes()) {
            size_t j_idx = 0;
            for (NodeId j : c.Nodes()) {
                if (i != j) {
                    if (c.ExchangeCapacity(i_idx, j_idx) > 0)
                        BOOST_REQUIRE_LE(dis[i], dis[j]+1);
                }
                j_idx++;
            }
            i_idx++;
        }
    }
}

void TestExcessBounds(const SubmodularFlow& sf) {
    const auto& dis = sf.GetDis();
    const auto& excess = sf.GetExcess();
    const NodeId s = sf.GetS();
    const int max_active = sf.GetMaxActive();
    const int min_active = sf.GetMinActive();

    for (NodeId i = 0; i < sf.GetNumNodes(); ++i) {
        if (excess[i] > 0 && dis[i] <= dis[s]) {
            BOOST_REQUIRE_LE(dis[i], max_active);
            BOOST_REQUIRE_GE(dis[i], min_active);
        }
    }
}

void TestInvariants(const SubmodularFlow& sf) {
    TestNonnegativeCapacities(sf);
    TestExcess(sf);
    TestDistance(sf);
    TestExcessBounds(sf);
}

void TestFinalInvariants(const SubmodularFlow& sf) {
    const auto& excess = sf.GetExcess();
    const auto& dis = sf.GetDis();
    const NodeId s = sf.GetS();
    for (NodeId i = 0; i < sf.GetNumNodes(); ++i) {
        if (dis[i] <= dis[s])
            BOOST_REQUIRE_EQUAL(excess[i], 0);
    }
}

void TestInvariantsPreserved(SubmodularFlow& sf) {
    sf.PushRelabelInit();
    TestInvariants(sf);

    while (sf.PushRelabelNotDone()) {
        sf.PushRelabelStep();
        TestInvariants(sf);
    }
    TestFinalInvariants(sf);
}

BOOST_AUTO_TEST_CASE(minimalGraph) {
    SubmodularFlow sf;
    SetupMinimalFlow(sf);
    TestInvariantsPreserved(sf);
}

BOOST_AUTO_TEST_CASE(largeGraph) {
    SubmodularFlow sf;

    const size_t n = 100;
    const size_t k = 4;
    const size_t m = 100;
    const REAL clique_range = 100;
    const REAL unary_mean = 800;
    const REAL unary_var = 1600;
    const unsigned int seed = 0;

    GenRandom(sf, n, k, m, clique_range, unary_mean, unary_var, seed);
    TestInvariantsPreserved(sf);
}

BOOST_AUTO_TEST_CASE(computeMinCut) {
    SubmodularFlow sf;

    const size_t n = 10;
    const size_t k = 4;
    const size_t m = 10;
    const REAL clique_range = 100;
    const REAL unary_mean = 800;
    const REAL unary_var = 1600;
    const unsigned int seed = 0;

    GenRandom(sf, n, k, m, clique_range, unary_mean, unary_var, seed);

    sf.PushRelabel();
    const auto& dis = sf.GetDis();
    const NodeId s = sf.GetS();
    std::vector<int> pr_labels;
    for (size_t i = 0; i < n; ++i) {
        if (dis[i] < dis[s])
            pr_labels.push_back(0);
        else
            pr_labels.push_back(1);
    }

    sf.ComputeMinCut();

    BOOST_CHECK_EQUAL(sf.ComputeEnergy(), sf.ComputeEnergy(pr_labels));
    for (size_t i = 0; i < n; ++i) {
        BOOST_CHECK_EQUAL(sf.GetLabel(i), pr_labels[i]);
    }

    const auto& cliques = sf.GetCliques();
    for (auto cp : cliques) {
        BOOST_CHECK_EQUAL(cp->ComputeEnergyAlpha(pr_labels), 0);
    }
}

BOOST_AUTO_TEST_SUITE_END()




BOOST_AUTO_TEST_SUITE(flowTests)

/* Sanity check to make sure basic flow computation working on a minimally
 * sized graph.
 */
BOOST_AUTO_TEST_CASE(minimalFlow) {
    SubmodularFlow sf;
    SetupMinimalFlow(sf);

    sf.PushRelabel();
    sf.ComputeMinCut();

    // Minimum energy is -1 with all 4 nodes labeled 1. Check that this
    // was the answer we found.
    BOOST_CHECK_EQUAL(sf.ComputeEnergy(), -1);
    for (NodeId i = 0; i < 4; ++i) {
        BOOST_CHECK_EQUAL(sf.GetLabel(i), 1);
    }
}

/* More complicated test case on a larger graph.
 *
 * GenRandom generates a random submodular function that can be turned into
 * a submodular quadratic function by HigherOrderEnergy. We generate the
 * same energy function for both SubmodularFlow and HigherOrderEnergy
 * and then check that they give the same answer.
 */
BOOST_AUTO_TEST_CASE(identicalToHigherOrder) {
    SubmodularFlow sf;
    HigherOrderEnergy<REAL, 4> ho;

    const size_t n = 100;
    const size_t k = 4;
    const size_t m = 100;
    const REAL clique_range = 100;
    const REAL unary_mean = 800;
    const REAL unary_var = 1600;
    const unsigned int seed = 0;

    GenRandom(sf, n, k, m, clique_range, unary_mean, unary_var, seed);
    GenRandom(ho, n, k, m, clique_range, unary_mean, unary_var, seed);

    sf.PushRelabel();
    sf.ComputeMinCut();

    QPBO<REAL> qr(n, 0);
    ho.ToQuadratic(qr);
    qr.Solve();

    std::vector<int> qr_labels;
    for (size_t i = 0; i < n; ++i)
        qr_labels.push_back(qr.GetLabel(i));

    // If this test fails, there's a problem in the higher-order code. Email me
    BOOST_CHECK_EQUAL(sf.ComputeEnergy(qr_labels)*2, qr.ComputeTwiceEnergy());

    BOOST_CHECK_EQUAL(sf.ComputeEnergy()*2, qr.ComputeTwiceEnergy());
    for (size_t i = 0; i < n; ++i) {
        BOOST_REQUIRE(qr.GetLabel(i) >= 0);
        BOOST_REQUIRE_EQUAL(sf.GetLabel(i), qr.GetLabel(i));
    }
}

BOOST_AUTO_TEST_SUITE_END()

