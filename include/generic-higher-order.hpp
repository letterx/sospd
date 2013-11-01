#ifndef _GENERIC_HIGHER_ORDER_HPP_
#define _GENERIC_HIGHER_ORDER_HPP_

#include "higher-order-energy.hpp"
#include "HOCR.h"
#include <string>

enum class OptType {
    Fix,
    Fix_I, // FGBZ with QPBOI
    HOCR,
    GRD,
    GRD_Heur,
    PC,
    PC_I, // PairwiseCover with QPBOI
    PC_Grid,
    PC_Grid_I,
    Grad
};

inline std::string ToString(OptType ot) {
    switch (ot) {
        case OptType::Fix: return "fix";
        case OptType::Fix_I: return "fix-i";
        case OptType::HOCR: return "hocr";
        case OptType::GRD: return "grd";
        case OptType::GRD_Heur: return "grd-heur";
        case OptType::PC: return "pc";
        case OptType::PC_I: return "pc-i";
        case OptType::PC_Grid: return "pc-grid";
        case OptType::PC_Grid_I: return "pc-grid-i";
        case OptType::Grad: return "grad";
        default: return "unknown";
    }
}

template <typename Optimizer>
void AddVars(Optimizer& opt, size_t numVars) {
    opt.AddVars(numVars);
}

template <typename REAL, int D>
void AddVars(PBF<REAL, D>& opt, size_t numVars) {
    // noop
}


template <typename Optimizer, typename Energy>
void AddConstantTerm(Optimizer& opt, Energy r) {
    // noop
}

template <typename Optimizer, typename Energy>
void AddUnaryTerm(Optimizer& opt, int v, Energy coeff) {
    opt.AddUnaryTerm(v, coeff);
}

template <typename REAL, int D>
void AddUnaryTerm(PBF<REAL, D>& opt, int v, REAL coeff) {
    opt.AddUnaryTerm(v, 0, coeff);
}



template <typename Optimizer, typename Energy>
void AddClique(Optimizer& opt, int d, const Energy *coeffs, const int *vars) {
    std::vector<int> vec_vars(vars, vars+d);
    std::vector<Energy> vec_coeffs(coeffs, coeffs+(1 << d));
    opt.AddClique(vec_vars, vec_coeffs);
}

template <typename REAL, int D>
void AddClique(PBF<REAL, D>& opt, int d, const REAL *coeffs, const int *vars) {
    opt.AddHigherTerm(d, const_cast<int*>(vars), const_cast<REAL*>(coeffs));
}

template <typename Opt, typename QR>
void ToQuadratic(Opt& opt, QR& qr) {
    opt.ToQuadratic(qr);
}

template <typename REAL, int D, typename QR>
void ToQuadratic(PBF<REAL, D>& opt, QR& qr) {
    PBF<REAL, 2> tmp_qr;
    opt.toQuadratic(tmp_qr);
    convert(qr, tmp_qr);
}

#endif
