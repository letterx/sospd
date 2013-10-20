#ifndef _IMAGE_HPP_
#define _IMAGE_HPP_
/*
 * image.hpp
 *
 * Copyright 2012 Alexander Fix
 * See LICENSE.txt for license information
 */

#include <boost/shared_array.hpp>

/*
 * A simple image class. Each Image<T> is a wrapper around a smart-pointer
 * to the actual data, so copy-assignment is cheap, but shares data.
 */
template <typename T>
class Image {
    public:
        Image();
        Image(int height, int width);
        Image(const Image<T>& i);
        Image& operator=(const Image<T>& i);
        // Copy does a deep copy of the image data.
        void Copy(const Image<T>& i);


        // Indexing through operator() is clamped, so negative indices, and 
        // indices greater than {height|width} are clamped to be inside the
        // image boundaries
        const T& operator()(int row, int col) const;
        T& operator()(int row, int col);

        // At is not bounds checked, hence faster
        T& At(int i) { return _data[i]; }
        const T& At(int i) const { return _data[i]; }

        int Height() const { return _height; }
        int Width() const { return _width; }

        T* Data() { return _data.get(); }
        const T* Data() const { return _data.get(); }

    private:
        friend void ImageToFile(const Image<unsigned char>& im, const char *filename);
        friend Image<unsigned char> ImageFromFile(const char *filename);
        boost::shared_array<T> _data;
        int _height;
        int _width;
};

Image<unsigned char> ImageFromFile(const char *filename);
void WriteToFile(const Image<unsigned char>& im, const char *filename);

typedef Image<unsigned char> Image_uc;

template <typename T>
inline Image<T>::Image()
    : _data(), _height(0), _width(0)
{ }

template <typename T>
inline Image<T>::Image(int height, int width)
    : _data(), _height(height), _width(width)
{
    _data = boost::shared_array<T>(new T[height*width]);
    for (int i = 0; i < height*width; ++i) {
        _data[i] = 0;
    }
}

template <typename T>
inline Image<T>::Image(const Image& i)
    : _data(i._data), _height(i._height), _width(i._width)
{ }

template <typename T>
inline Image<T>& Image<T>::operator=(const Image& i) {
    _data = i._data;
    _width = i._width;
    _height = i._height;
    return *this;
}

template <typename T>
inline void Image<T>::Copy(const Image& i) {
    _width = i._width;
    _height = i._height;
    _data = boost::shared_array<T> (new T[_height*_width]);
    for (int index = 0; index < _width*_height; ++index) {
        _data[index] = i._data[index];
    }
}

template <typename T>
inline const T& Image<T>::operator()(int row, int col) const {
    if (row >= _height) row = _height-1;
    if (row < 0) row = 0;
    if (col >= _width) col = _width-1;
    if (col < 0) col = 0;
    return _data[row*_width + col];
}

template <typename T>
inline T& Image<T>::operator()(int row, int col) {
    if (row >= _height) row = _height-1;
    if (row < 0) row = 0;
    if (col >= _width) col = _width-1;
    if (col < 0) col = 0;
    return _data[row*_width + col];
}
#endif
