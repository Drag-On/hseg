//
// Created by jan on 12.01.17.
//

#include <iostream>
#include <helper/opencv_helper.h>
#include "Image/FeatureImage.h"
#include "matio.h"

FeatureImage::FeatureImage(Coord width, Coord height, Coord dim)
    : m_width(width),
      m_height(height),
      m_dim(dim)
{
    m_features.resize(width * height, Feature::Zero(dim));
}

FeatureImage::FeatureImage(std::string const& filename)
{
    read(filename);
}

bool FeatureImage::read(std::string const& filename)
{
    mat_t* matfp = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
    if (matfp == nullptr)
    {
        std::cerr << "Error opening feature file \"" << filename << "\"." << std::endl;
        return false;
    }

    matvar_t *matvar = Mat_VarReadInfo(matfp,"features");
    if ( matvar == nullptr )
    {
        std::cerr << "Error finding feature map in file \"" << filename << "\"." << std::endl;
        Mat_Close(matfp);
        return false;
    }
    else
    {
        // Read in dimensions of the image and of the features
        m_height = (Coord) matvar->dims[0];
        m_width = (Coord) matvar->dims[1];
        m_dim = (Coord) matvar->dims[2];

        Mat_VarReadDataAll(matfp, matvar);

        if(matvar->data_type != MAT_T_SINGLE)
        {
            std::cerr << "Wrong datatype!" << std::endl;
            Mat_VarFree(matvar);
            Mat_Close(matfp);
            return false;
        }

        m_features.reserve(m_width * m_height);

        for(size_t y = 0; y < m_height; ++y)
        {
            for(size_t x = 0; x < m_width; ++x)
            {
                Feature f = Feature::Zero(m_dim);
                for(size_t c = 0; c < m_dim; ++c)
                {
                    float data = ((float*)matvar->data)[y + x * m_height + c * m_height* m_width];
                    f(c) = data;
                }
                m_features.push_back(f);
            }
        }

        Mat_VarFree(matvar);
    }

    Mat_Close(matfp);
    return true;
}

bool FeatureImage::write(std::string const& filename)
{
    mat_t* matfp = Mat_Create(filename.c_str(), nullptr);
    if (matfp == nullptr)
    {
        std::cerr << "Error creating feature file \"" << filename << "\"." << std::endl;
        return false;
    }

    size_t dims[] = {m_height, m_width, m_dim};
    float* data = new float[m_height * m_width * m_dim];
    matvar_t *matvar = Mat_VarCreate("features", MAT_C_SINGLE, MAT_T_SINGLE, 3, dims, data, MAT_F_DONT_COPY_DATA);
    if ( matvar == nullptr )
    {
        std::cerr << "Error creating feature map in file \"" << filename << "\"." << std::endl;
        Mat_Close(matfp);
        delete [] data;
        return false;
    }
    else
    {
        // Write features to mat variable
        for(size_t y = 0; y < m_height; ++y)
        {
            for(size_t x = 0; x < m_width; ++x)
            {
                for(size_t c = 0; c < m_dim; ++c)
                {
                    ((float*)matvar->data)[y + x * m_height + c * m_height* m_width] = at(x, y)[c];
                }
            }
        }
        Mat_VarWrite(matfp, matvar, MAT_COMPRESSION_NONE);
        Mat_VarFree(matvar);
    }

    Mat_Close(matfp);
    delete [] data;
    return true;
}

Coord FeatureImage::width() const
{
    return m_width;
}

Coord FeatureImage::height() const
{
    return m_height;
}

Coord FeatureImage::dim() const
{
    return m_dim;
}

Feature const& FeatureImage::at(Coord x, Coord y) const
{
    assert(x + y * m_width < m_features.size());
    return m_features[x + y * m_width];
}

Feature& FeatureImage::at(Coord x, Coord y)
{
    assert(x + y * m_width < m_features.size());
    return m_features[x + y * m_width];
}

Feature const& FeatureImage::atSite(SiteId i) const
{
    assert(i < m_features.size());
    return m_features[i];
}

Feature& FeatureImage::atSite(SiteId i)
{
    assert(i < m_features.size());
    return m_features[i];
}

std::vector<Feature>& FeatureImage::data()
{
    return m_features;
}

std::vector<Feature> const& FeatureImage::data() const
{
    return m_features;
}

void FeatureImage::subtract(FeatureImage const& other)
{
    assert(other.m_features.size() == m_features.size());

    for(size_t i = 0; i < m_features.size(); ++i)
        m_features[i] -= other.m_features[i];
}

FeatureImage::operator cv::Mat() const
{
    cv::Mat result(m_height, m_width, helper::opencv::getOpenCvType<float>(m_dim));

    for (Coord y = 0; y < m_height; ++y)
    {
        for (Coord x = 0; x < m_width; ++x)
        {
            for (Coord c = 0; c < m_dim; ++c)
                ((float*)result.data)[(x+y*m_width) * m_dim + c] = at(x, y)[c];
        }
    }

    return result;
}

void FeatureImage::rescale(float factor, bool interpolate)
{
    cv::Mat img = static_cast<cv::Mat>(*this);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(), factor, factor, interpolate ? cv::INTER_LINEAR : cv::INTER_NEAREST);

    m_width = static_cast<size_t>(resized.cols);
    m_height = static_cast<size_t>(resized.rows);
    m_features.resize(m_width * m_height, Feature::Zero(m_dim));

    for (Coord y = 0; y < m_height; ++y)
    {
        for (Coord x = 0; x < m_width; ++x)
        {
            for (Coord c = 0; c < m_dim; ++c)
                at(x, y)[c] = ((float*)resized.data)[resized.cols * y * m_dim + x * m_dim + c];
        }
    }
}

void FeatureImage::rescale(Coord width, Coord height, bool interpolate)
{
    cv::Mat img = static_cast<cv::Mat>(*this);
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(width, height), 0, 0, interpolate ? cv::INTER_LINEAR : cv::INTER_NEAREST);

    m_width = static_cast<size_t>(resized.cols);
    m_height = static_cast<size_t>(resized.rows);
    m_features.resize(m_width * m_height, Feature::Zero(m_dim));

    for (Coord y = 0; y < m_height; ++y)
    {
        for (Coord x = 0; x < m_width; ++x)
        {
            for (Coord c = 0; c < m_dim; ++c)
                at(x, y)[c] = ((float*)resized.data)[resized.cols * y * m_dim + x * m_dim + c];
        }
    }
}

void FeatureImage::minMax(float* min, float* max) const
{
    if(min != nullptr)
        *min = std::numeric_limits<float>::max();
    if(max != nullptr)
        *max = std::numeric_limits<float>::min();

    for (Coord y = 0; y < m_height; ++y)
    {
        for (Coord x = 0; x < m_width; ++x)
        {
            for (Coord c = 0; c < m_dim; ++c)
            {
                float const f = at(x, y)[c];
                if(min != nullptr && f < *min)
                    *min = f;
                if(max != nullptr && f > *max)
                    *max = f;
            }
        }
    }
}

void FeatureImage::flipHorizontally()
{
    for (Coord y = 0; y < m_height; ++y)
    {
        for (Coord x = 0; x < std::floor(m_width / 2.f); ++x)
        {
            Feature f = at(x, y);
            at(x, y) = at(m_width - x - 1, y);
            at(m_width - x - 1, y) = f;
        }
    }
}

void FeatureImage::addFrom(FeatureImage const& other, int x, int y, int w, int h)
{
    for(int d_x = x; d_x < m_width && d_x < x + w; ++d_x)
    {
        for(int d_y = y; d_y < m_height && d_y < y + h; ++d_y)
        {
            at(d_x, d_y) += other.at(d_x - x, d_y - y);
        }
    }
}

void FeatureImage::normalize()
{
    for (SiteId i = 0; i < m_width * m_height; ++i)
    {
        float sum = atSite(i).sum();
        if (sum > 0)
            atSite(i) /= sum;
    }
}
