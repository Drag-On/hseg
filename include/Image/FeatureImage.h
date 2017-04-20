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

    void minMax(float* min, float* max) const;

    void flipHorizontally();

    /**
     * Copy features from \p other to destination, adding them to old values. Ignores features that would be out of bounds.
     * @param other Features to copy from
     * @param x Destination x
     * @param y Destination y
     * @param w Width
     * @param h Height
     */
    void addFrom(FeatureImage const& other, int x, int y, int w, int h);

    /**
     * Normalizes the feature map such that every feature sums up to one
     */
    void normalize();

protected:
    Coord m_width;
    Coord m_height;
    Coord m_dim;
    std::vector<Feature> m_features;
};


#endif //HSEG_FEATUREIMAGE_H
