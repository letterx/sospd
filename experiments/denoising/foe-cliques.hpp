#ifndef _FOE_CLIQUES_HPP_
#define _FOE_CLIQUES_HPP_
/*
 * foe-cliques.hpp
 *
 * Copyright 2012 Alexander Fix
 * See LICENSE.txt for license information
 *
 * Example of deriving from CliqueEnergy (defined in clique.hpp) to implement 
 * the Field of Experts energy used for the denoising algorithm
 */

#include "clique.hpp"
#include <math.h>
#include <iostream>

typedef int REAL;
const double DoubleToREAL = 1000;

/*
 * The Field of Experts energy, defined for a 2x2 patch of the image.
 * Note that the only thing we really need to override from the abstract 
 * base class is operator(), which actually calculates the FoE energy of a 
 * 2x2 patch.
 */
class FoEEnergy : public CliqueEnergy<REAL, unsigned char, 4> {
    public:
        FoEEnergy(int size, int nbd[])
            : CliqueEnergy<REAL, unsigned char, 4>(size, nbd) { }
        virtual REAL operator()(const unsigned char buf[]) const;
};

/*
 * The unary energy is defined for a single pixel. It penalizes the squared
 * distance from the original observed value. Note that we've added a new data
 * member _orig, which keeps track of the originally observed value. 
 */
class FoEUnaryEnergy : public CliqueEnergy<REAL, unsigned char, 4> {
    public:
        FoEUnaryEnergy(int *index, unsigned char originalImagePixel) 
            : CliqueEnergy<REAL, unsigned char, 4>(1, index),
              _orig(originalImagePixel) { }

        virtual REAL operator()(const unsigned char buf[]) const;
        static double sigma;

    private:
        unsigned char _orig;
};

#endif
