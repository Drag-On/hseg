//
// Created by jan on 18.08.16.
//

#ifndef HSEG_IMAGE_H
#define HSEG_IMAGE_H

#include <cstddef>
#include <vector>
#include <string>
#include <helper/opencv_helper.h>
#include "Coordinates.h"
#include "typedefs.h"
#include <cv.hpp>
#if CV_MAJOR_VERSION==3
#include <opencv2/imgcodecs.hpp>
#else
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#endif

using ImgCoords = Coords2d<Coord>;


/**
 * Image of variable size and with a variable amount of channels
 * @tparam T Value type
 * @tparam C Amount of channels
 */
template<typename T, size_t C>
class Image
{
public:
    /**
     * Default constructor
     */
    Image() = default;

    /**
     * Construct an empty image
     * @param width Image width
     * @param height Image height
     */
    Image(Coord width, Coord height) noexcept;

    /**
     * Copy constructor
     * @param other Image to copy
     */
    Image(Image const& other) = default;

    /**
     * Move constructor
     * @param other Image to move
     */
    Image(Image&& other) = default;

    /**
     * Construct image from open cv matrix
     * @details If the given matrix doesn't fit, this will create an empty image
     * @param mat Matrix
     */
    Image(cv::Mat const& mat) noexcept;

    /**
     * Destructor
     */
    ~Image() = default;

    /**
     * Copy-assignment
     * @param other Image to copy
     * @return Reference to this
     */
    Image& operator=(Image const& other) = default;

    /**
     * Move-assignment
     * @param other Image to move
     * @return Reference to this
     */
    Image& operator=(Image&& other) = default;

    /**
     * Assign data from open cv matrix
     * @details Will not change the underlying image if the passed argument doesn't fit
     * @param mat Matrix
     * @return Reference to this
     */
    Image& operator=(cv::Mat const& mat) noexcept;

    /**
     * Cast to cv::Mat
     * @return The image
     */
    explicit operator cv::Mat() const;

    /**
     * Compares two images for equality
     * @param other Other image
     * @return True if images are identical, otherwise false
     */
    bool operator==(Image const& other) const;

    /**
     * Read an image from file
     * @param filename File to read from
     * @return True in case of success, otherwise false
     */
    bool read(std::string const& filename);

    /**
     * Write an image to file
     * @param filename File to write to
     * @return True in case of success, otherwise false
     */
    bool write(std::string const& filename);

    /**
     * Converts the image to CieLab in the floating point range. Note that this is better for the CieLab color space:
     * While it can be represented in bytes, this messes up its euclidean distance. In floating point notation the
     * distances are fine.
     * @return CieLab color image
     * @note This works only when the image is in BGR color space
     */
    Image<float, C> getCieLabImg() const;

    /**
     * @return Image width
     */
    Coord width() const;

    /**
     * @return Image height
     */
    Coord height() const;

    /**
     * @return Amount of channels
     */
    Coord channels() const;

    /**
     * @return Amount of pixels
     */
    SiteId pixels() const;

    /**
     * Computes the amount of pixels that are different from another image of the same size
     * @param other Other image
     * @return Amount of differing pixels
     */
    size_t diff(Image<T, C> const& other) const;

    /**
     * Retrieve a pixel value
     * @param x X coordinate
     * @param y Y coordinate
     * @param c Channel
     * @return Pixel value at the given pixel and channel
     */
    T const& at(Coord x, Coord y, Coord c = 0) const;

    /**
     * Retrieve a pixel value
     * @param x X coordinate
     * @param y Y coordinate
     * @param c Channel
     * @return Pixel value at the given pixel and channel
     */
    T& at(Coord x, Coord y, Coord c = 0);

    /**
     * Retrieve a pixel value
     * @param site Site id
     * @param c Channel
     * @return Pixel value at the given pixel and channel
     */
    T const& atSite(SiteId site, Coord c = 0) const;

    /**
     * Retrieve a pixel value
     * @param site Site id
     * @param c Channel
     * @return Pixel value at the given pixel and channel
     */
    T& atSite(SiteId site, Coord c = 0);

    /**
     * Rescales the image
     * @param factor Factor
     * @param interpolate Enables or disables interpolation
     */
    void rescale(float factor, bool interpolate = false);

    /**
     * Rescales the image
     * @param width New width
     * @param height New height
     * @param interpolate Enables or disables interpolation
     */
    void rescale(Coord width, Coord height, bool interpolate = false);

    /**
     * @return Minimum and maximum value
     */
    std::pair<T, T> minMax() const;

    /**
     * Stretches the image's colors such that the smalles value coincides with a given minimum, and its largest value
     * conincides with a given maximum
     * @param min New minimum value
     * @param max New maximum value
     */
    Image& scaleColorSpace(T min, T max);

    /**
     * Access the base data
     * @return Data vector
     */
    std::vector<T> const& data() const;

    /**
     * Access the base data
     * @return Data vector
     */
    std::vector<T>& data();

private:
    Coord m_width = 0;
    Coord m_height = 0;
    std::vector<T> m_data;
};

/**
 * 3-Channel color image
 */
template<typename T>
using ColorImage = Image<T, 3>;

/**
 * RGB image encoded as bytes
 */
using RGBImage = ColorImage<unsigned char>;

/**
 * CieLab image encoded as floating points
 */
using CieLabImage = ColorImage<float>;

/**
 * Label image encoded as bytes
 */
using LabelImage = Image<Label, 1>;


template<typename T, size_t C>
Image<T, C>::Image(Coord width, Coord height) noexcept
        : m_width(width),
          m_height(height),
          m_data(width * height * C, 0)
{
}

template<typename T, size_t C>
Image<T, C>::Image(cv::Mat const& mat) noexcept
{
    if (!mat.data || mat.channels() != C)
        return;

    m_width = mat.cols;
    m_height = mat.rows;
    m_data.resize(m_width * m_height * C, 0);

    for (size_t y = 0; y < m_height; ++y)
    {
        for (size_t x = 0; x < m_width; ++x)
        {
            auto color = mat.at<cv::Vec<T, C>>(y, x);
            for (size_t c = 0; c < C; ++c)
                at(x, y, c) = color[c];
        }
    }
}

template<typename T, size_t C>
Image<T, C>& Image<T, C>::operator=(cv::Mat const& mat) noexcept
{
    *this = Image<T, C>(mat);
    return *this;
}

template<typename T, size_t C>
bool Image<T, C>::read(std::string const& filename)
{
    cv::Mat mat = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    if (!mat.data)
        return false;

    if (mat.channels() != C)
        return false;

    m_width = static_cast<size_t>(mat.cols);
    m_height = static_cast<size_t>(mat.rows);
    m_data.resize(m_width * m_height * C, 0);

    for (size_t y = 0; y < m_height; ++y)
    {
        for (size_t x = 0; x < m_width; ++x)
        {
            auto color = mat.at<cv::Vec<uchar, C>>(y, x);
            for (size_t c = 0; c < C; ++c)
                at(x, y, c) = color[c];
        }
    }

    return true;
}

template<typename T, size_t C>
bool Image<T, C>::write(std::string const& filename)
{
    cv::Mat img = static_cast<cv::Mat>(*this);
    return cv::imwrite(filename, img);
}

template<typename T, size_t C>
Image<T, C>::operator cv::Mat() const
{
    cv::Mat result(m_height, m_width, helper::opencv::getOpenCvType<T>(C));

    for (Coord y = 0; y < m_height; ++y)
    {
        for (Coord x = 0; x < m_width; ++x)
        {
            cv::Vec<T, C> color;
            for (Coord c = 0; c < C; ++c)
                color[c] = at(x, y, c);
            result.at<cv::Vec<T, C>>(y, x) = color;
        }
    }

    return result;
}

template<typename T, size_t C>
bool Image<T, C>::operator==(Image<T, C> const& other) const
{
    return m_data == other.m_data;
}

template<typename T, size_t C>
inline Coord Image<T, C>::width() const
{
    return m_width;
}

template<typename T, size_t C>
inline Coord Image<T, C>::height() const
{
    return m_height;
}

template<typename T, size_t C>
inline Coord Image<T, C>::channels() const
{
    return C;
}

template<typename T, size_t C>
inline SiteId Image<T, C>::pixels() const
{
    return m_width * m_height;
}

template<typename T, size_t C>
inline T const& Image<T, C>::at(Coord x, Coord y, Coord c) const
{
    assert(c < C);
    assert(x + (y * m_width) + (c * m_width * m_height) < m_data.size());
    return m_data[x + (y * m_width) + (c * m_width * m_height)];
}

template<typename T, size_t C>
inline T& Image<T, C>::at(Coord x, Coord y, Coord c)
{
    assert(c < C);
    assert(x + (y * m_width) + (c * m_width * m_height) < m_data.size());
    return m_data[x + (y * m_width) + (c * m_width * m_height)];
}

template<typename T, size_t C>
inline T const& Image<T, C>::atSite(SiteId site, Coord c) const
{
    assert(c < C);
    assert(site + (c * m_width * m_height) < m_data.size());
    return m_data[site + (c * m_width * m_height)];
}

template<typename T, size_t C>
inline T& Image<T, C>::atSite(SiteId site, Coord c)
{
    assert(c < C);
    assert(site + (c * m_width * m_height) < m_data.size());
    return m_data[site + (c * m_width * m_height)];
}

template<typename T, size_t C>
Image<float, C> Image<T, C>::getCieLabImg() const
{
    cv::Mat mat = static_cast<cv::Mat>(*this);
    cv::Mat floatMat;
    mat.convertTo(floatMat, CV_32FC(C));
    floatMat /= 255;
    cv::Mat floatMatCieLab;
    cv::cvtColor(floatMat, floatMatCieLab, CV_BGR2Lab);
    Image<float, C> img(floatMatCieLab);
    return img;
}

template<typename T, size_t C>
std::pair<T, T> Image<T, C>::minMax() const
{
    T min = std::numeric_limits<T>::max();
    T max = std::numeric_limits<T>::min();
    for (auto const& e : m_data)
    {
        if (e < min)
            min = e;
        else if (e > max)
            max = e;
    }
    return std::pair<T, T>(min, max);
}

template<typename T, size_t C>
Image<T, C>& Image<T, C>::scaleColorSpace(T min, T max)
{
    auto minMax = this->minMax();
    for (auto& px : m_data)
        px = ((max - min) * (px - minMax.first)) / (minMax.second - minMax.first) + min;

    return *this;
}

template<typename T, size_t C>
size_t Image<T, C>::diff(Image<T, C> const& other) const
{
    assert(other.width() == width() && other.height() == height());

    size_t result = 0;
    for (SiteId i = 0; i < pixels(); ++i)
        for (Coord c = 0; c < C; ++c)
            if (atSite(i, c) != other.atSite(i, c))
                result++;
    return result;
}

template<typename T, size_t C>
void Image<T, C>::rescale(float factor, bool interpolate)
{
    cv::Mat img = static_cast<cv::Mat>(*this);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(), factor, factor, interpolate ? cv::INTER_LINEAR : cv::INTER_NEAREST);

    m_width = static_cast<size_t>(resized.cols);
    m_height = static_cast<size_t>(resized.rows);
    m_data.resize(m_width * m_height * C, 0);

    for (Coord y = 0; y < m_height; ++y)
    {
        for (Coord x = 0; x < m_width; ++x)
        {
            auto color = resized.at<cv::Vec<uchar, C>>(y, x);
            for (Coord c = 0; c < C; ++c)
                at(x, y, c) = color[c];
        }
    }
}

template<typename T, size_t C>
void Image<T, C>::rescale(Coord width, Coord height, bool interpolate)
{
    cv::Mat img = static_cast<cv::Mat>(*this);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(width, height), 0, 0, interpolate ? cv::INTER_LINEAR : cv::INTER_NEAREST);

    m_width = static_cast<size_t>(resized.cols);
    m_height = static_cast<size_t>(resized.rows);
    m_data.resize(m_width * m_height * C, 0);

    for (Coord y = 0; y < m_height; ++y)
    {
        for (Coord x = 0; x < m_width; ++x)
        {
            auto color = resized.at<cv::Vec<T, C>>(y, x);
            for (Coord c = 0; c < C; ++c)
                at(x, y, c) = color[c];
        }
    }
}

template<typename T, size_t C>
std::vector<T> const& Image<T, C>::data() const
{
    return m_data;
}

template<typename T, size_t C>
std::vector<T>& Image<T, C>::data()
{
    return m_data;
}

#endif //HSEG_IMAGE_H
