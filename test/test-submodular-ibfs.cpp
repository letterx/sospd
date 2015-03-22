#include <boost/test/unit_test.hpp>
#include <iostream>
#include "submodular-ibfs.hpp"
#include "higher-order-energy.hpp"
#include "gen-random.hpp"
#include "qpbo.hpp"

typedef SubmodularIBFS::NodeId NodeId;

/* Sets up a submodular flow problem with a single clique. The clique has 4
* nodes, and is equal to -1 when all 4 nodes are set to 1, 0 otherwise.
*/
static void SetupMinimalFlow(SubmodularIBFS& sf) {

    sf.AddNode(4);

    std::vector<NodeId> nodes = {0, 1, 2, 3};
    const size_t numAssignments = 1 << 4;
    std::vector<REAL> energyTable(numAssignments, 0);
    energyTable[0xf] = -1;
    sf.AddClique(nodes, energyTable);
}

void TestConstructor(SubmodularIBFS& sf) {
    BOOST_CHECK_EQUAL(sf.GetConstantTerm(), 0);
    BOOST_CHECK_EQUAL(sf.Graph().NumNodes(), 0);
    BOOST_CHECK_EQUAL(sf.Graph().GetC_si().size(), 0);
    BOOST_CHECK_EQUAL(sf.Graph().GetC_it().size(), 0);
    BOOST_CHECK_EQUAL(sf.Graph().GetPhi_si().size(), 0);
    BOOST_CHECK_EQUAL(sf.Graph().GetPhi_it().size(), 0);
    BOOST_CHECK_EQUAL(sf.GetLabels().size(), 0);
    BOOST_CHECK_EQUAL(sf.Graph().GetNumCliques(), 0);
    BOOST_CHECK_EQUAL(sf.Graph().GetCliques().size(), 0);
    BOOST_CHECK_EQUAL(sf.Graph().GetNeighbors().size(), 0);
    BOOST_CHECK_EQUAL(sf.ComputeEnergy(), 0);
}

void TestMinimalFlowSetup(SubmodularIBFS& sf) {
    SetupMinimalFlow(sf);

    BOOST_CHECK_EQUAL(sf.GetConstantTerm(), 0);
    BOOST_CHECK_EQUAL(sf.Graph().GetC_si().size(), 4);
    BOOST_CHECK_EQUAL(sf.Graph().GetC_it().size(), 4);
    BOOST_CHECK_EQUAL(sf.Graph().GetPhi_si().size(), 4);
    BOOST_CHECK_EQUAL(sf.Graph().GetPhi_it().size(), 4);
    BOOST_CHECK_EQUAL(sf.GetLabels().size(), 4);
    BOOST_CHECK_EQUAL(sf.Graph().GetNumCliques(), 1);
    BOOST_CHECK_EQUAL(sf.Graph().GetCliques().size(), 1);
    BOOST_CHECK_EQUAL(sf.Graph().GetNeighbors().size(), 4);
    BOOST_CHECK_EQUAL(sf.ComputeEnergy(), 0);

    sf.Graph().ResetFlow();
    sf.Graph().UpperBoundCliques(SoSGraph::UBfn::cvpr14);

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

void TestRandomFlowSetup(SubmodularIBFS& sf) {
    const size_t n = 100;
    const size_t k = 4;
    const size_t m = 100;
    const REAL clique_range = 100;
    const REAL unary_mean = 800;
    const REAL unary_var = 1600;
    const unsigned int seed = 0;

    GenRandom(sf, n, k, m, clique_range, unary_mean, unary_var, seed);

    BOOST_CHECK_EQUAL(sf.Graph().GetC_si().size(), n);
    BOOST_CHECK_EQUAL(sf.Graph().GetC_it().size(), n);
    BOOST_CHECK_EQUAL(sf.Graph().GetPhi_si().size(), n);
    BOOST_CHECK_EQUAL(sf.Graph().GetPhi_it().size(), n);
    BOOST_CHECK_EQUAL(sf.GetLabels().size(), n);
    BOOST_CHECK_EQUAL(sf.Graph().GetNumCliques(), m);
    BOOST_CHECK_EQUAL(sf.Graph().GetCliques().size(), m);
    BOOST_CHECK_EQUAL(sf.Graph().GetNeighbors().size(), n);

}

/* Check that for a clique c, the energy is always >= 0, and is equal to 0 at
* the all 0 and all 1 labelings.
*/
void CheckNormalized(const SoSGraph::IBFSEnergyTableClique& c, std::vector<int>& labels) {
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
        if (assgn == 0 || assgn == max_assgn-1)
            BOOST_REQUIRE_EQUAL(c.ComputeAlphaEnergy(labels), 0);
        else if (assgn == max_assgn - 1)
            BOOST_REQUIRE_EQUAL(c.ComputeAlphaEnergy(labels), 0);
        else
            BOOST_REQUIRE_GE(c.ComputeAlphaEnergy(labels), 0);
    }
}

/* Check that all source-sink capacities are >= 0, and that cliques
* are normalized (i.e., are >= 0 for all labelings)
*/
void TestRandomFlowNormalized(SubmodularIBFS& sf) {
    const size_t n = 100;
    const size_t k = 4;
    const size_t m = 100;
    const REAL clique_range = 100;
    const REAL unary_mean = 800;
    const REAL unary_var = 1600;
    const unsigned int seed = 0;

    GenRandom(sf, n, k, m, clique_range, unary_mean, unary_var, seed);
    sf.Graph().UpperBoundCliques(SoSGraph::UBfn::cvpr14);

    for (const auto& c : sf.Graph().GetCliques()) {
        BOOST_CHECK_EQUAL(c.Nodes().size(), 4);
        CheckNormalized(c, sf.GetLabels());
    }
}

static void CheckCut(SubmodularIBFS& sf) {
    auto& phi_si = sf.Graph().GetPhi_si();
    auto& phi_it = sf.Graph().GetPhi_it();
    auto& c_si = sf.Graph().GetC_si();
    auto& c_it = sf.Graph().GetC_it();
    for (SubmodularIBFS::NodeId i = 0; i < sf.Graph().NumNodes(); ++i) {
        int label = sf.GetLabel(i);
        if (label == 0) {
            BOOST_CHECK_EQUAL(phi_si[i], c_si[i]);
            if (phi_si[i] != c_si[i])
                std::cout << "Bad source arc to " << i << "\n";
        } else {
            BOOST_CHECK_EQUAL(phi_it[i], c_it[i]);
            if (phi_it[i] != c_it[i])
                std::cout << "Bad sink arc to " << i << "\n";
        }
        for (auto arc = sf.Graph().ArcsBegin(i); arc != sf.Graph().ArcsEnd(i); ++arc) {
            auto j = arc.Target();
            if (j >= sf.Graph().NumNodes())
                continue;
            int label_j = sf.GetLabel(j);
            if (label == 1 && label_j == 0) {
                BOOST_CHECK_EQUAL(sf.Graph().ResCap(arc, true), 0);
                if (sf.Graph().ResCap(arc, true) != 0)
                    std::cout << "Bad Arc: " << arc.Source() << ", " 
                        << arc.Target() << "\t"
                        << "Clique: " << arc.cliqueId() << "\n";
            }
        }
    }
    for (auto& c : sf.Graph().GetCliques()) {
        auto reparamEnergy = c.ComputeAlphaEnergy(sf.GetLabels());
        BOOST_CHECK_EQUAL(reparamEnergy, 0);
        int assgn = 0;
        REAL sumPhi = 0;
        for (size_t i = 0; i < c.Size(); ++i) {
            if (sf.GetLabel(c.Nodes()[i]) == 1) {
                sumPhi += c.AlphaCi()[i];
                assgn |= (1 << i);
            }
        }
        BOOST_CHECK_EQUAL(sumPhi, c.EnergyTable()[assgn]);
    }
}

/* Sanity check to make sure basic flow computation working on a minimally
* sized graph.
*/
void TestMinimalFlow(SubmodularIBFS& sf) {
    SetupMinimalFlow(sf);

    sf.Solve();
    CheckCut(sf);

    // Minimum energy is -1 with all 4 nodes labeled 1. Check that this
    // was the answer we found.
    BOOST_CHECK_EQUAL(sf.ComputeEnergy(), -1);
    for (NodeId i = 0; i < 4; ++i) {
        BOOST_CHECK_EQUAL(sf.GetLabel(i), 1);
    }
}

void TestSearchSource(SubmodularIBFS& crf) {
	crf.AddNode(3);
	crf.AddUnaryTerm(0, 12, 6);
	crf.AddUnaryTerm(1, 8, 8);
	crf.AddUnaryTerm(2, 6, 12);
	NodeId node_array[3] = {0, 1, 2};
	std::vector<NodeId> node(node_array, node_array + 3);
	REAL energy_array[8] = {0, 3, 1, 2, 0, 2, 0, 0};
	std::vector<REAL> energy(energy_array, energy_array + 8);
	crf.AddClique(node, energy);
	crf.Solve();
    CheckCut(crf);
    BOOST_CHECK_EQUAL(crf.GetLabel(0), 1);
    BOOST_CHECK_EQUAL(crf.GetLabel(1), 1);
    BOOST_CHECK_EQUAL(crf.GetLabel(2), 0);
    BOOST_CHECK_EQUAL(crf.ComputeEnergy(), 22);

    std::vector<int> label;
    label.push_back(1);
    label.push_back(0);
    label.push_back(0);
    BOOST_CHECK_EQUAL(crf.ComputeEnergy(label), 23);
}

/* More complicated test case on a larger graph.
*
* GenRandom generates a random submodular function that can be turned into
* a submodular quadratic function by HigherOrderEnergy. We generate the
* same energy function for both SubmodularIBFS and HigherOrderEnergy
* and then check that they give the same answer.
*/
void TestIdenticalToHigherOrder(SubmodularIBFS& sf) {
    HigherOrderEnergy<REAL, 4> ho;

    const size_t n = 16000;
    const size_t m = 16000;
    const size_t k = 4;
    const REAL clique_range = 100;
    const REAL unary_mean = 800;
    const REAL unary_var = 1600;
    const unsigned int seed = 0;

    GenRandom(sf, n, k, m, clique_range, unary_mean, unary_var, seed);
    GenRandom(ho, n, k, m, clique_range, unary_mean, unary_var, seed);

    sf.Solve();
    CheckCut(sf);

    QPBO<REAL> qr(n, 0);
    ho.ToQuadratic(qr);
    qr.Solve();

    std::vector<int> qr_labels;
    for (size_t i = 0; i < n; ++i)
        qr_labels.push_back(qr.GetLabel(i));

    SubmodularIBFS orig;
    GenRandom(orig, n, k, m, clique_range, unary_mean, unary_var, seed);
    orig.Graph().ResetFlow();
    sf.Graph().UpperBoundCliques(SoSGraph::UBfn::cvpr14);
    for (size_t i = 0; i < n; ++i) {
        BOOST_REQUIRE_EQUAL(sf.Graph().GetC_si()[i], orig.Graph().GetC_si()[i]);
        BOOST_REQUIRE_EQUAL(sf.Graph().GetC_it()[i], orig.Graph().GetC_it()[i]);
    }
    for (size_t i = 0; i < m; ++i) {
        auto& c1 = sf.Graph().GetCliques()[i];
        auto& c2 = orig.Graph().GetCliques()[i];
        for (int assgn = 0; assgn < (1 << k); ++assgn) {
            BOOST_REQUIRE_EQUAL(c1.EnergyTable()[assgn], c2.EnergyTable()[assgn]);
        }
    }



    /*
     *std::cout << "sf(qr_labels): " << sf.ComputeEnergy(qr_labels)*2 << "\n";
     *std::cout << "orig(qr_labels): " << orig.ComputeEnergy(qr_labels)*2 << "\n";
     *std::cout << "qr(qr_labels): " << qr.ComputeTwiceEnergy() << "\n";
     *std::cout << "sf(sf_labels): " << sf.ComputeEnergy()*2 << "\n";
     *std::cout << "Checking " << n << " variables\n";
     */
    int misses = 0;
    for (size_t i = 0; i < n; ++i) {
        if (sf.GetLabel(i) != qr.GetLabel(i)) {
            std::cout << "Different at " << i << "\n";
            std::cout << sf.GetLabel(i) << " vs " << qr.GetLabel(i) << "\n";
            misses++;
            if (misses == 10)
                break;
        }
    }
    // If this test fails, there's a problem in the higher-order code. Email me
    BOOST_CHECK_EQUAL(sf.ComputeEnergy(qr_labels)*2, qr.ComputeTwiceEnergy());

    BOOST_CHECK_EQUAL(sf.ComputeEnergy()*2, qr.ComputeTwiceEnergy());
    for (size_t i = 0; i < n; ++i) {
        BOOST_REQUIRE(qr.GetLabel(i) >= 0);
        BOOST_WARN_EQUAL(sf.GetLabel(i), qr.GetLabel(i));
    }
}

BOOST_AUTO_TEST_SUITE(TestBidirectional)
    BOOST_AUTO_TEST_CASE(Constructor) {
        SubmodularIBFS sf;
        TestConstructor(sf);
    }
    BOOST_AUTO_TEST_CASE(MinimalFlowSetup) {
        SubmodularIBFS sf;
        TestMinimalFlowSetup(sf);
    }
    BOOST_AUTO_TEST_CASE(RandomFlowSetup) {
        SubmodularIBFS sf;
        TestRandomFlowSetup(sf);
    }
    BOOST_AUTO_TEST_CASE(RandomFlowNormalized) {
        SubmodularIBFS sf;
        TestRandomFlowNormalized(sf);
    }
    BOOST_AUTO_TEST_CASE(MinimalFlow) {
        SubmodularIBFS sf;
        TestMinimalFlow(sf);
    }
    BOOST_AUTO_TEST_CASE(SearchSource) {
        SubmodularIBFS sf;
        TestSearchSource(sf);
    }
    BOOST_AUTO_TEST_CASE(IdenticalToHigherOrder) {
        SubmodularIBFS sf;
        TestIdenticalToHigherOrder(sf);
    }
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(TestSource)
    BOOST_AUTO_TEST_CASE(Constructor) {
        SubmodularIBFSParams params{ SubmodularIBFSParams::FlowAlgorithm::source };
        SubmodularIBFS sf {params};
        TestConstructor(sf);
    }
    BOOST_AUTO_TEST_CASE(MinimalFlowSetup) {
        SubmodularIBFSParams params{ SubmodularIBFSParams::FlowAlgorithm::source };
        SubmodularIBFS sf {params};
        TestMinimalFlowSetup(sf);
    }
    BOOST_AUTO_TEST_CASE(RandomFlowSetup) {
        SubmodularIBFSParams params{ SubmodularIBFSParams::FlowAlgorithm::source };
        SubmodularIBFS sf {params};
        TestRandomFlowSetup(sf);
    }
    BOOST_AUTO_TEST_CASE(RandomFlowNormalized) {
        SubmodularIBFSParams params{ SubmodularIBFSParams::FlowAlgorithm::source };
        SubmodularIBFS sf {params};
        TestRandomFlowNormalized(sf);
    }
    BOOST_AUTO_TEST_CASE(MinimalFlow) {
        SubmodularIBFSParams params{ SubmodularIBFSParams::FlowAlgorithm::source };
        SubmodularIBFS sf {params};
        TestMinimalFlow(sf);
    }
    BOOST_AUTO_TEST_CASE(SearchSource) {
        SubmodularIBFSParams params{ SubmodularIBFSParams::FlowAlgorithm::source };
        SubmodularIBFS sf {params};
        TestSearchSource(sf);
    }
    BOOST_AUTO_TEST_CASE(IdenticalToHigherOrder) {
        SubmodularIBFSParams params{ SubmodularIBFSParams::FlowAlgorithm::source };
        SubmodularIBFS sf {params};
        TestIdenticalToHigherOrder(sf);
    }
BOOST_AUTO_TEST_SUITE_END()
