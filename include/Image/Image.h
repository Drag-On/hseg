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
#include <helper/opencv_helper.h>

enum class ColorSpace
{
    BGR,
    CieLab,
};

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
     * @param colorSpace Color space the matrix is in
     */
    Image(cv::Mat const& mat, ColorSpace colorSpace = ColorSpace::BGR) noexcept;

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
     * Read an image from file
     * @param filename File to read from
     * @return True in case of success, otherwise false
     */
    bool read(std::string const& filename);

    /**
     * Convert image to another color space
     */
    void convertTo(ColorSpace space);

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
     * @return Color space of the image
     */
    ColorSpace colorSpace() const;

    /**
     * Retrieve a pixel value
     * @param x X coordinate
     * @param y Y coordinate
     * @param c Channel
     * @return Pixel value at the given pixel and channel
     */
    T const& at(int x, int y, int c = 0) const;

    /**
     * Retrieve a pixel value
     * @param x X coordinate
     * @param y Y coordinate
     * @param c Channel
     * @return Pixel value at the given pixel and channel
     */
    T& at(int x, int y, int c = 0);

    /**
     * Retrieve a pixel value
     * @param site Site id
     * @param c Channel
     * @return Pixel value at the given pixel and channel
     */
    T const& atSite(size_t site, int c = 0) const;

    /**
     * Retrieve a pixel value
     * @param site Site id
     * @param c Channel
     * @return Pixel value at the given pixel and channel
     */
    T& atSite(size_t site, int c = 0);

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

private:
    int m_width;
    int m_height;
    std::vector<T> m_data;
    ColorSpace m_colorSpace = ColorSpace::BGR;
};

/**
 * 3-Channel color image
 */
template <typename T>
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
Image<T, C>::Image(cv::Mat const& mat, ColorSpace colorSpace) noexcept
{
    if (!mat.data || mat.channels() != C)
        return;

    m_width = mat.cols;
    m_height = mat.rows;
    m_data.resize(m_width * m_height * C, 0);

    for (int y = 0; y < m_height; ++y)
    {
        for (int x = 0; x < m_width; ++x)
        {
            auto color = mat.at<cv::Vec<T, C>>(y, x);
            for (int c = 0; c < C; ++c)
                at(x, y, c) = color[c];
        }
    }
    m_colorSpace = colorSpace;
}

template<typename T, int C>
Image<T, C>& Image<T, C>::operator=(cv::Mat const& mat) noexcept
{
    *this = Image<T, C>(mat);
    return *this;
}

template<typename T, int C>
bool Image<T, C>::read(std::string const& filename)
{
    cv::Mat mat = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    if (!mat.data)
        return false;

    if (mat.channels() != C)
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
    cv::Mat result(m_height, m_width, helper::opencv::getOpenCvType<T>(C));

    for (int y = 0; y < m_height; ++y)
    {
        for (int x = 0; x < m_width; ++x)
        {
            cv::Vec<T, C> color;
            for (int c = 0; c < C; ++c)
                color[c] = at(x, y, c);
            result.at<cv::Vec<T, C>>(y, x) = color;
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
ColorSpace Image<T, C>::colorSpace() const
{
    return m_colorSpace;
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
T const& Image<T, C>::atSite(size_t site, int c) const
{
    assert(c < C);
    return m_data[site + (c * m_width * m_height)];
}

template<typename T, int C>
T& Image<T, C>::atSite(size_t site, int c)
{
    assert(c < C);
    return m_data[site + (c * m_width * m_height)];
}

template<typename T, int C>
void Image<T, C>::convertTo(ColorSpace space)
{
    if (m_colorSpace != space)
    {
        cv::Mat mat = static_cast<cv::Mat>(*this);
        switch (space)
        {
            case ColorSpace::BGR:
                cv::cvtColor(mat, mat, CV_Lab2BGR);
                break;
            case ColorSpace::CieLab:
                cv::cvtColor(mat, mat, CV_BGR2Lab);
                break;
        }
        *this = mat;
    }
}

template<typename T, int C>
Image<float, C> Image<T, C>::getCieLabImg() const
{
    assert(colorSpace() == ColorSpace::BGR);

    cv::Mat mat = static_cast<cv::Mat>(*this);
    cv::Mat floatMat;
    mat.convertTo(floatMat, CV_32FC(C));
    floatMat /= 255;
    cv::Mat floatMatCieLab;
    cv::cvtColor(floatMat, floatMatCieLab, CV_BGR2Lab);
    Image<float, C> img(floatMatCieLab, ColorSpace::CieLab);
    return img;
}

template<typename T, int C>
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

template<typename T, int C>
Image<T, C>& Image<T, C>::scaleColorSpace(T min, T max)
{
    auto minMax = this->minMax();
    for (auto& px : m_data)
        px = ((max - min) * (px - minMax.first)) / (minMax.second - minMax.first) + min;

    return *this;
}

#endif //HSEG_IMAGE_H
