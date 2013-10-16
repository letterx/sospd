#ifndef _SUBMODULAR_FUNCTIONS_HPP_
#define _SUBMODULAR_FUNCTIONS_HPP_

#include <vector>
#include <cstdint>

typedef int REAL;
typedef uint32_t Assgn;

std::vector<REAL> SubmodularUpperBound(int n, const std::vector<REAL>& energyTable);
std::vector<REAL> SubmodularLowerBound(int n, const std::vector<REAL>& energyTable);

// Takes in a set s (given by bitstring) and returns new energy such that
// f(t | s) = f(t) for all t. Does not change f(t) for t disjoint from s
// I.e., creates a set s whose members have zero marginal gain for all t
std::vector<REAL> ZeroMarginalSet(int n, const std::vector<REAL>& energyTable, Assgn s);

bool CheckSubmodular(int n, const std::vector<REAL>& energyTable);
bool CheckUpperBoundInvariants(int n, const std::vector<REAL>& energyTable,
        const std::vector<REAL>& upperBound);

#endif
