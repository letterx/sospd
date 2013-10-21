#ifndef _SUBMODULAR_FUNCTIONS_HPP_
#define _SUBMODULAR_FUNCTIONS_HPP_

#include "energy-common.hpp"
#include <vector>
#include <cstdint>

typedef uint32_t Assgn;

void SubmodularUpperBound(int n, std::vector<REAL>& energyTable);
REAL SubmodularLowerBound(int n, std::vector<REAL>& energyTable, bool early_finish = false);

// Takes in a set s (given by bitstring) and returns new energy such that
// f(t | s) = f(t) for all t. Does not change f(t) for t disjoint from s
// I.e., creates a set s whose members have zero marginal gain for all t
void ZeroMarginalSet(int n, std::vector<REAL>& energyTable, Assgn s);

// Updates f to f'(S) = f(S) + psi(S)
void AddLinear(int n, std::vector<REAL>& energyTable, const std::vector<REAL>& psi);

// Updates f to f'(S) = f(S) - psi1(S) - psi2(V\S)
void SubtractLinear(int n, std::vector<REAL>& energyTable, 
        const std::vector<REAL>& psi1, const std::vector<REAL>& psi2);

// Modifies an energy function to be >= 0, with f(0) = f(V) = 0
// energyTable is modified in place, must be submodular
// psi must be length n, gets filled so that 
//  f'(S) = f(S) + psi(S)
// where f' is the new energyTable, and f is the old one
void Normalize(int n, std::vector<REAL>& energyTable, std::vector<REAL>& psi);

bool CheckSubmodular(int n, const std::vector<REAL>& energyTable);
bool CheckUpperBoundInvariants(int n, const std::vector<REAL>& energyTable,
        const std::vector<REAL>& upperBound);

#endif
