//
// Created by jan on 12.01.17.
//

#ifndef HSEG_FEATUREIMAGE_H
#define HSEG_FEATUREIMAGE_H

#include <string>
#include <vector>
#include <typedefs.h>
#include <opencv2/opencv.hpp>
#include "Feature.h"

class FeatureImage
{
public:
    FeatureImage() = default;

    FeatureImage(Coord width, Coord height, Coord dim);

    FeatureImage(std::string const& filename);

    bool read(std::string const& filename);

    bool write(std::string const& filename);

    Coord width() const;

    Coord height() const;

    /**
     * Provides the feature dimensionality
     * @return
     */
    Coord dim() const;

    Feature const& at(Coord x, Coord y) const;

    Feature& at(Coord x, Coord y);

    Feature const& atSite(SiteId i) const;

    Feature& atSite(SiteId i);

    std::vector<Feature>& data();

    std::vector<Feature> const& data() const;

    void subtract(FeatureImage const& other);

    explicit operator cv::Mat() const;

    void minMax(float* min, float* max) const;

protected:
    Coord m_width;
    Coord m_height;
    Coord m_dim;
    std::vector<Feature> m_features;
};


#endif //HSEG_FEATUREIMAGE_H
