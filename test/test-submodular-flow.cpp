#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE SubmodularFlow
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "submodular-flow.hpp"

BOOST_AUTO_TEST_SUITE(basicSubmodularFlow)

BOOST_AUTO_TEST_CASE(sanityCheck) {
    BOOST_CHECK_EQUAL(2+2, 4);
}

BOOST_AUTO_TEST_SUITE_END()

