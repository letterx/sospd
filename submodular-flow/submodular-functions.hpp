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

/********************** Implementation *************************/

static inline Assgn NextPerm(Assgn v) {
    Assgn t = v | (v - 1); // t gets v's least significant 0 bits set to 1
    // Next set to 1 the most significant bit to change,
    // set to 0 the least significant ones, and add the necessary 1 bits.
    return (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctz(v) + 1));
}

inline void SubmodularUpperBound(int n, std::vector<REAL>& energyTable) {
    ASSERT(n < 32);
    int max_assgn = 1 << n;
    std::vector<REAL> oldEnergy(energyTable);
    std::vector<REAL> diffEnergy(max_assgn, 0);

    SubmodularLowerBound(n, energyTable);

    for (int i = 0; i < max_assgn; ++i)
        diffEnergy[i] = oldEnergy[i] - energyTable[i];
    for (int i = max_assgn - 2; i > 0; --i) {
        for (int k = 0; k < n; ++k) {
            REAL tmp = (diffEnergy[i | (1 << k)] + 1) / 2;
            if (tmp > diffEnergy[i])
                diffEnergy[i] = tmp;
        }
    }
    for (int i = 0; i < max_assgn; ++i) {
        energyTable[i] += diffEnergy[i];
    }
    
    if (!CheckSubmodular(n, energyTable)) SubmodularUpperBound(n, energyTable);
}

inline void OldSubmodularUpperBound(int n, std::vector<REAL>& energyTable) {
    ASSERT(n < 32);
    REAL max_energy = 0;
    REAL zero_energy = energyTable[0];
    REAL last_energy = energyTable.back();
    for (std::size_t i = 0; i < energyTable.size(); ++i) {
        max_energy = std::max(max_energy, energyTable[i]);
    }
    
    REAL max_diff = SubmodularLowerBound(n, energyTable);
    //ASSERT(CheckSubmodular(n, energyTable));

    for (REAL& e : energyTable) {
        e += max_diff;
    }
    energyTable[0] = zero_energy;
    energyTable.back() = last_energy;

    // Truncate singelton sets to max_energy, and then find a new lower-bound
    // (guaranteed to still upper-bound original function)
    for (int k = 0; k < n; ++k) {
        energyTable[1<<k] = std::min(energyTable[1<<k], max_energy);
    }
    SubmodularLowerBound(n, energyTable);

    //ASSERT(CheckSubmodular(n, energyTable));
}

inline REAL SubmodularLowerBound(int n, std::vector<REAL>& energyTable, bool early_finish) {
    ASSERT(n < 32);
    Assgn max_assgn = 1 << n;
    ASSERT(energyTable.size() == max_assgn);
    REAL max_diff = 0;

    // Need to iterate over all k bit subsets in increasing k
    for (int k = 1; k <= n; ++k) {
        bool changed = false;
        Assgn bound;
        if (k == 0) bound = 0;
        else bound = max_assgn - 1;
        Assgn s = (1 << k) - 1;
        do {
            REAL subtract_from_s = 0;
            for (int i = 0; i < n; ++i) {
                Assgn s_i = s ^ (1 << i); // Set s - i
                if (s_i >= s) continue;
                for (int j = i+1; j < n; ++j) {
                    Assgn s_j = s ^ (1 << j); // Set s - j
                    if (s_j >= s) continue;
                    Assgn s_ij = s_i & s_j;
                    REAL submodularity = energyTable[s] + energyTable[s_ij]
                        - energyTable[s_i] - energyTable[s_j];
                    if (submodularity > subtract_from_s) {
                        subtract_from_s = submodularity;
                    }
                }
            }
            energyTable[s] -= subtract_from_s;
            changed |= (subtract_from_s > 0);
            max_diff = std::max(max_diff, subtract_from_s);
            s = NextPerm(s);
        } while (s < bound);
        if (early_finish && !changed)
            break;
    }
    return max_diff;
}

inline void ZeroMarginalSet(int n, std::vector<REAL>& energyTable, Assgn s) {
    Assgn base_set = (1 << n) - 1;
    Assgn not_s = base_set & (~s);
    for (Assgn t = 0; t <= base_set; ++t)
        energyTable[t] = energyTable[t & not_s];
}

inline void AddLinear(int n, std::vector<REAL>& energyTable, const std::vector<REAL>& psi) {
    Assgn max_assgn = 1 << n;
    ASSERT(max_assgn == energyTable.size());
    ASSERT(n == int(psi.size()));
    REAL sum = 0;
    Assgn last_gray = 0;
    for (Assgn a = 1; a < max_assgn; ++a) {
        Assgn gray = a ^ (a >> 1);
        Assgn diff = gray ^ last_gray;
        int changed_bit = __builtin_ctz(diff);
        if (gray & diff)
            sum += psi[changed_bit];
        else
            sum -= psi[changed_bit];
        energyTable[gray] += sum;
        last_gray = gray;
    }
}

inline void SubtractLinear(int n, std::vector<REAL>& energyTable, 
        const std::vector<REAL>& psi1, const std::vector<REAL>& psi2) {
    Assgn max_assgn = 1 << n;
    ASSERT(max_assgn == energyTable.size());
    ASSERT(n == int(psi1.size()));
    ASSERT(n == int(psi2.size()));
    REAL sum = 0;
    for (int i = 0; i < n; ++i)
        sum += psi2[i];
    energyTable[0] -= sum;
    Assgn last_gray = 0;
    for (Assgn a = 1; a < max_assgn; ++a) {
        Assgn gray = a ^ (a >> 1);
        Assgn diff = gray ^ last_gray;
        int changed_idx = __builtin_ctz(diff);
        if (gray & diff)
            sum += psi1[changed_idx] - psi2[changed_idx];
        else
            sum += psi2[changed_idx] - psi1[changed_idx];
        energyTable[gray] -= sum;
        last_gray = gray;
    }
}

inline void Normalize(int n, std::vector<REAL>& energyTable, std::vector<REAL>& psi) {
    Assgn max_assgn = 1 << n;
    ASSERT(max_assgn == energyTable.size());
    ASSERT(n == int(psi.size()));
    ASSERT(energyTable[0] == 0);
    Assgn last_assgn = 0;
    Assgn this_assgn = 0;
    for (int i = 0; i < n; ++i) {
        this_assgn |= (1 << i);
        psi[i] = energyTable[last_assgn] - energyTable[this_assgn];
        last_assgn = this_assgn;
    }
    AddLinear(n, energyTable, psi);

    for (REAL e : energyTable)
        ASSERT(e >= 0);
    ASSERT(energyTable[0] == 0);
    ASSERT(energyTable[max_assgn-1] == 0);
}

inline bool CheckSubmodular(int n, const std::vector<REAL>& energyTable) {
    ASSERT(n < 32);
    Assgn max_assgn = 1 << n;
    ASSERT(energyTable.size() == max_assgn);

    for (Assgn s = 0; s < max_assgn; ++s) {
        for (int i = 0; i < n; ++i) {
            Assgn s_i = s | (1 << i);
            if (s_i == s) continue;
            for (int j = i+1; j < n; ++j) {
                Assgn s_j = s | (1 << j);
                if (s_j == s) continue;
                Assgn s_ij = s_i | s_j;

                REAL submodularity = energyTable[s] + energyTable[s_ij]
                    - energyTable[s_i] - energyTable[s_j];
                if (submodularity > 0) {
                    //std::cout << "Nonsubmodular: (" << s << ", " << i << ", " << j << "): ";
                    //std::cout << energyTable[s] << " " << energyTable[s_i] << " "
                    //    << energyTable[s_j] << " " << energyTable[s_ij] << "\n";
                    return false;
                }
            }
        }
    }
    return true;
}

inline bool CheckUpperBoundInvariants(int n, const std::vector<REAL>& energyTable,
        const std::vector<REAL>& upperBound) {
    int energy_len = energyTable.size();
    ASSERT(energy_len == int(upperBound.size()));
    REAL max_energy = std::numeric_limits<REAL>::min();
    for (int i = 0; i < energy_len; ++i) {
        if (energyTable[i] > upperBound[i])
            return false;
        max_energy = std::max(energyTable[i], max_energy);
    }
    for (int i = 0; i < n; ++i) {
        if (upperBound[1 << i] > max_energy)
            return false;
    }
    return CheckSubmodular(n, upperBound);
}

#endif
