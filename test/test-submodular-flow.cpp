#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SubmodularFlow
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "submodular-flow.hpp"
#include "higher-order.hpp"
#include "gen-random.hpp"
#include "QPBO.h"

BOOST_AUTO_TEST_SUITE(basicSubmodularFlow)

BOOST_AUTO_TEST_CASE(sanityCheck) {
    typedef SubmodularFlow::NodeId NodeId;

    SubmodularFlow sf;
    sf.AddNode(4);

    std::vector<NodeId> nodes = {0, 1, 2, 3};
    const size_t numAssignments = 1 << 4;
    std::vector<REAL> energyTable(numAssignments, 0);
    energyTable[0xf] = -1;
    sf.AddClique(nodes, energyTable);

    sf.PushRelabel();
    sf.ComputeMinCut();

    BOOST_CHECK_EQUAL(sf.ComputeEnergy(), -1);
    for (NodeId i = 0; i < 4; ++i) {
        BOOST_CHECK_EQUAL(sf.GetLabel(i), 1);
    }
}

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

    BOOST_CHECK_EQUAL(sf.ComputeEnergy()*2, qr.ComputeTwiceEnergy());
    for (size_t i = 0; i < n; ++i) {
        BOOST_REQUIRE(qr.GetLabel(i) >= 0);
        BOOST_REQUIRE_EQUAL(sf.GetLabel(i), qr.GetLabel(i));
    }
}

BOOST_AUTO_TEST_SUITE_END()

