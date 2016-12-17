//
// Created by jan on 18.08.16.
//

#ifndef HSEG_FEATURE_H
#define HSEG_FEATURE_H

#include <Image/Image.h>
#include <helper/coordinate_helper.h>
#include <Eigen/Dense>

/**
 * Represents a feature taken from a pixel. This is continuous and therefore has a mean.
 */
class Feature
{
private:
    Eigen::Matrix<Cost, 5, 1> m_features;
public:
    /**
     * Default constructor
     */
    Feature();

    /**
     * Construct a feature from an image at a certain site
     * @param color RGB image
     * @param site Site
     */
    template<typename T>
    Feature(Image<T, 3> const& color, SiteId site)
    {
        auto coords = helper::coord::siteTo2DCoordinate(site, color.width());
        m_features << color.atSite(site, 0), color.atSite(site, 1), color.atSite(site, 2), coords.x(), coords.y();
    }

    /**
     * Adds features component-wise
     * @param other Feature to add
     * @return Reference to this
     */
    inline Feature& operator+=(Feature const& other)
    {
        m_features += other.m_features;
        return *this;
    }

    /**
     * Adds features component-wise
     * @param other Feature to add
     * @return The new feature
     */
    inline Feature operator+(Feature const& other) const
    {
        Feature f = *this;
        f += other;
        return f;
    }

    /**
     * Subtracts features component-wise
     * @param other Feature to subtract
     * @return Reference to this
     */
    inline Feature& operator-=(Feature const& other)
    {
        m_features -= other.m_features;
        return *this;
    }

    /**
     * Subtracts features component-wise
     * @param other Feature to subtract
     * @return The new feature
     */
    inline Feature operator-(Feature const& other) const
    {
        Feature f = *this;
        f -= other;
        return f;
    }

    /**
     * Unary minus, inverts all components
     * @return Negative feature
     */
    inline Feature operator-() const
    {
        Feature f = *this;
        f -= Feature();
        return f;
    }

    /**
     * Divides all components by the same number
     * @param count Denominator
     * @return Reference to this
     */
    inline Feature& operator/=(size_t count)
    {
        m_features /= static_cast<Cost>(count);
        return *this;
    }

    /**
     * Divides all components by the same number
     * @param count Denominator
     * @return The new feature
     */
    inline Feature operator/(size_t count) const
    {
        Feature f = *this;
        f /= count;
        return f;
    }

    /**
     * Squares all elements of this feature
     */
    inline void squareElements()
    {
        m_features = m_features.cwiseProduct(m_features);
    }

    /**
     * @return Feature with all elements squared
     */
    inline Feature getSquaredElements() const
    {
        Feature f = *this;
        f.squareElements();
        return f;
    }

    /**
     * @return X coordinate
     */
    inline Cost x() const
    {
        return m_features(3);
    }

    /**
     * @return Y coordinate
     */
    inline Cost y() const
    {
        return m_features(4);
    }

    /**
     * @return R component
     */
    inline Cost r() const
    {
        return m_features(0);
    }

    /**
     * @return G component
     */
    inline Cost g() const
    {
        return m_features(1);
    }

    /**
     * @return B component
     */
    inline Cost b() const
    {
        return m_features(2);
    }

    inline Eigen::Matrix<Cost, 5, 1> const& vec() const
    {
        return m_features;
    }
};

#endif //HSEG_FEATURE_H
