//
// Created by jan on 18.08.16.
//

#ifndef HSEG_IMAGE_H
#define HSEG_IMAGE_H

#include <cstddef>
#include <vector>
#include <string>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgcodecs/imgcodecs_c.h>
#include <cv.hpp>

/**
 * Image of variable size and with a variable amount of channels
 * @tparam T Value type
 * @tparam C Amount of channels
 */
template<typename T, int C>
class Image
{
public:
    /**
     * Default constructor
     */
    Image() noexcept = default;

    /**
     * Construct an empty image
     * @param width Image width
     * @param height Image height
     */
    Image(int width, int height) noexcept;

    /**
     * Copy constructor
     * @param other Image to copy
     */
    Image(Image const& other) noexcept = default;

    /**
     * Move constructor
     * @param other Image to move
     */
    Image(Image&& other) noexcept = default;

    /**
     * Destructor
     */
    ~Image() noexcept = default;

    /**
     * Copy-assignment
     * @param other Image to copy
     * @return Reference to this
     */
    Image& operator=(Image const& other) noexcept = default;

    /**
     * Move-assignment
     * @param other Image to move
     * @return Reference to this
     */
    Image& operator=(Image&& other) noexcept = default;

    /**
     * Cast to cv::Mat
     * @return The image
     */
    explicit operator cv::Mat() const;

    /**
     * Read an image from file
     * @param filename File to read from
     * @return True in case of success, otherwise false
     */
    bool read(std::string const& filename);

    /**
     * @return Image width
     */
    int width() const;

    /**
     * @return Image height
     */
    int height() const;

    /**
     * @return Amount of channels
     */
    int channels() const;

    /**
     * @return Amount of pixels
     */
    size_t pixels() const;

    /**
     * Retrieve a pixel value
     * @param x X coordinate
     * @param y Y coordinate
     * @param c Channel
     * @return Pixel value at the given pixel and channel
     */
    T const& at(int x, int y, int c) const;

    /**
     * Retrieve a pixel value
     * @param x X coordinate
     * @param y Y coordinate
     * @param c Channel
     * @return Pixel value at the given pixel and channel
     */
    T& at(int x, int y, int c);

    /**
     * Retrieve a pixel value
     * @param site Site id
     * @param c Channel
     * @return Pixel value at the given pixel and channel
     */
    T const& at(size_t site, int c) const;

    /**
     * Retrieve a pixel value
     * @param site Site id
     * @param c Channel
     * @return Pixel value at the given pixel and channel
     */
    T& at(size_t site, int c);

private:
    int m_width;
    int m_height;
    std::vector<T> m_data;
};

/**
 * RGB image encoded as bytes
 */
using RGBImage = Image<unsigned char, 3>;

/**
 * A label
 */
using Label = unsigned int;

/**
 * Label image encoded as bytes
 */
using LabelImage = Image<Label, 1>;


template<typename T, int C>
Image<T, C>::Image(int width, int height) noexcept
        : m_width(width),
          m_height(height),
          m_data(width * height * C, 0)
{
}

template<typename T, int C>
bool Image<T, C>::read(std::string const& filename)
{
    cv::Mat mat = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    if (!mat.data)
        return false;

    if (!mat.channels() == C)
        return false;

    m_width = mat.cols;
    m_height = mat.rows;
    m_data.resize(m_width * m_height * C, 0);

    for (int y = 0; y < m_height; ++y)
    {
        for (int x = 0; x < m_width; ++x)
        {
            auto color = mat.at<cv::Vec<uchar, C>>(y, x);
            for (int c = 0; c < C; ++c)
                at(x, y, c) = color[c];
        }
    }

    return true;
}

template<typename T, int C>
Image<T, C>::operator cv::Mat() const
{
    cv::Mat result(m_height, m_width, CV_8UC(C));

    for (int y = 0; y < m_height; ++y)
    {
        for (int x = 0; x < m_width; ++x)
        {
            cv::Vec<uchar, C> color;
            for (int c = 0; c < C; ++c)
                color[c] = at(x, y, c);
            result.at<cv::Vec<uchar, C>>(y, x) = color;
        }
    }

    return result;
}

template<typename T, int C>
int Image<T, C>::width() const
{
    return m_width;
}

template<typename T, int C>
int Image<T, C>::height() const
{
    return m_height;
}

template<typename T, int C>
int Image<T, C>::channels() const
{
    return C;
}

template<typename T, int C>
size_t Image<T, C>::pixels() const
{
    return m_width * m_height;
}

template<typename T, int C>
T const& Image<T, C>::at(int x, int y, int c) const
{
    assert(c < C);
    return m_data[x + (y * m_width) + (c * m_width * m_height)];
}

template<typename T, int C>
T& Image<T, C>::at(int x, int y, int c)
{
    assert(c < C);
    return m_data[x + (y * m_width) + (c * m_width * m_height)];
}

template<typename T, int C>
T const& Image<T, C>::at(size_t site, int c) const
{
    assert(c < C);
    return m_data[site + (c * m_width * m_height)];
}

template<typename T, int C>
T& Image<T, C>::at(size_t site, int c)
{
    assert(c < C);
    return m_data[site + (c * m_width * m_height)];
}


#endif //HSEG_IMAGE_H
