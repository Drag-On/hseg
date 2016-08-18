//
// Created by jan on 18.08.16.
//

#ifndef HSEG_IMAGE_H
#define HSEG_IMAGE_H

#include <cstddef>
#include <vector>
#include <string>
#include <opencv2/imgcodecs.hpp>

/**
 * Image of variable size and with a variable amount of channels
 */
template<typename T>
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
     * @param channels Image channels
     */
    Image(int width, int height, int channels) noexcept;

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

private:
    int m_width;
    int m_height;
    int m_channels;
    std::vector<T> m_data;
};

template <typename T>
Image<T>::Image(int width, int height, int channels) noexcept
        : m_width(width),
          m_height(height),
          m_channels(channels),
          m_data(width * height * channels, 0)
{
}

template <typename T>
bool Image<T>::read(std::string const& filename)
{
    cv::Mat mat = cv::imread(filename);
    if (!mat.data)
        return false;

    m_width = mat.cols;
    m_height = mat.rows;
    m_channels = mat.channels();
    m_data.resize(m_width * m_height * m_channels, 0);

    for (int y = 0; y < m_width; ++y)
    {
        for (int x = 0; x < m_height; ++x)
        {
            for (int l = 0; l < m_channels; ++l)
            {
                at(x, y, l) = mat.at<char>(y, x, l);
            }
        }
    }

    return true;
}

template <typename T>
Image<T>::operator cv::Mat() const
{
    cv::Mat mat(m_height, m_width, CV_8UC(m_channels));
    for (int y = 0; y < m_width; ++y)
    {
        for (int x = 0; x < m_height; ++x)
        {
            for (int l = 0; l < m_channels; ++l)
            {
                mat.at<char>(y, x, l) = at(x, y, l);
            }
        }
    }
    return mat;
}

template <typename T>
int Image<T>::width() const
{
    return m_width;
}

template <typename T>
int Image<T>::height() const
{
    return m_height;
}

template <typename T>
int Image<T>::channels() const
{
    return m_channels;
}

template <typename T>
T const& Image<T>::at(int x, int y, int c) const
{
    return m_data[x + (y * m_width) + (c * m_width * m_height)];
}

template <typename T>
T& Image<T>::at(int x, int y, int c)
{
    return m_data[x + (y * m_width) + (c * m_width * m_height)];
}


#endif //HSEG_IMAGE_H
