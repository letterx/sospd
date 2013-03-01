#include "gen-random.hpp"
#include <random>
#include <algorithm>
#include "submodular-flow.hpp"
#include "higher-order.hpp"

template <typename REAL>
void GenRandomEnergyTable(std::vector<REAL>& energy_table, size_t k, REAL clique_range) {
    uint32_t num_assignments = 1 << k;
    energy_table = std::vector<REAL>(num_assignments, 0);

    // Implement me!
}

template <typename HigherOrder, typename REAL>
void GenRandom(HigherOrder& ho, 
        size_t n, 
        size_t k, 
        size_t m, 
        REAL clique_range, 
        REAL unary_mean,
        REAL unary_var,
        unsigned int seed)
{
    typedef typename HigherOrder::NodeId NodeId;
    std::mt19937 random_gen(seed); // Random number generator
    std::uniform_int_distribution<NodeId> node_gen(0, n-1);
    std::normal_distribution<double> unary_gen(unary_mean, unary_var);

    ho.AddNode(n);
    for (size_t i = 0; i < m; ++i) {
        std::vector<NodeId> clique_nodes;
        while (clique_nodes.size() < k) {
            NodeId new_node = node_gen(random_gen);
            if (std::count(clique_nodes.begin(), clique_nodes.end(), new_node) == 0)
                clique_nodes.push_back(new_node);
        }
        std::vector<REAL> energy_table;
        GenRandomEnergyTable(energy_table, k, clique_range);
        ho.AddClique(clique_nodes, energy_table);
    }

    for (size_t i = 0; i < n; ++i) {
        double unary_term = unary_gen(random_gen);
        ho.AddUnaryTerm(i, unary_term);
    }
}


void DummyInstantiateTemplates() {
    HigherOrderEnergy<REAL, 4> hoe;
    GenRandom(hoe, 0, 0, 0, (REAL)0, (REAL)0, (REAL)0, 0);

    SubmodularFlow sf;
    GenRandom(sf, 0, 0, 0, (REAL)0, (REAL)0, (REAL)0, 0);
}
