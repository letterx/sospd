/*
 * image.hpp
 *
 * Copyright 2012 Alexander Fix
 * See LICENSE.txt for license information
 */

#include "image.hpp"
#include <string.h>
#include <fstream>
#include <iostream>

/* 
 * Reads and writes Image<unsigned char> as a .pgm file
 */

Image_uc ImageFromFile(const char *filename)
{
    std::ifstream in(filename);
    char controlBuf[128];
    in.getline(controlBuf, 128);
    if (strncmp(controlBuf, "P5", 128) != 0) {
        std::cerr << "Magic number does not match PGM" << std::endl;
        exit(-1);
    }
    int width, height, max;
    in >> width;
    in >> height;
    in.ignore(1);
    in >> max;
    in.ignore(1);
    boost::shared_array<unsigned char> data(new unsigned char[width*height]);
    for (int i = 0; i < height; ++i) {
        in.read((char *)data.get() + i*width, width);
    }
    Image_uc i(height, width);
    i._data = data;
    return i;
}

void ImageToFile(const Image_uc& im, const char *filename) 
{
    std::ofstream out(filename);
    out << "P5\n";
    out << im._width << " " << im._height << '\n';
    out << 255 << '\n';
    for (int i = 0; i < im._height; ++i) {
        out.write((char *)im._data.get() + i*im._width, im._width);
    }
}
