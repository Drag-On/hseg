//
// Created by jan on 12.01.17.
//

#include <iostream>
#include "Image/FeatureImage.h"
#include "matio.h"

FeatureImage::FeatureImage(std::string const& filename)
{
    read(filename);
}

FeatureImage::FeatureImage(Coord width, Coord height, Coord dim)
        : m_width(width),
          m_height(height),
          m_dim(dim)
{
    m_features.resize(width * height * dim);
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
