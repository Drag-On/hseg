//
// Created by jan on 18.08.16.
//

#ifndef HSEG_FEATURE_H
#define HSEG_FEATURE_H

#include <Image/Image.h>
#include <helper/coordinate_helper.h>

/**
 * Represents a feature taken from a pixel. This is continuous and therefore has a mean.
 */
class Feature
{
public:
    /**
     * Default constructor
     */
    Feature() = default;

    /**
     * Construct a feature from an image at a certain site
     * @param color RGB image
     * @param site Site
     */
    template<typename T>
    Feature(Image<T, 3> const& color, size_t site)
    {
        auto coords = helper::coord::siteTo2DCoordinate(site, color.width());
        m_x = m_spatialWeight * coords.first;
        m_y = m_spatialWeight * coords.second;
        m_r = m_colorWeight * color.at(site, 0);
        m_g = m_colorWeight * color.at(site, 1);
        m_b = m_colorWeight * color.at(site, 2);
    }

    /**
     * Computes the squared euclidean distance to another feature
     * @param other Other feature
     * @return Distance
     */
    float sqDistanceTo(Feature const& other) const;

    /**
     * Adds features component-wise
     * @param other Feature to add
     * @return Reference to this
     */
    Feature& operator+=(Feature const& other);

    /**
     * Adds features component-wise
     * @param other Feature to add
     * @return The new feature
     */
    Feature operator+(Feature const& other) const;

    /**
     * Subtracts features component-wise
     * @param other Feature to subtract
     * @return Reference to this
     */
    Feature& operator-=(Feature const& other);

    /**
     * Subtracts features component-wise
     * @param other Feature to subtract
     * @return The new feature
     */
    Feature operator-(Feature const& other);

    /**
     * Unary minus, inverts all components
     * @return Negative feature
     */
    Feature operator-() const;

    /**
     * Divides all components by the same number
     * @param count Denominator
     * @return Reference to this
     */
    Feature& operator/=(size_t count);

    /**
     * Divides all components by the same number
     * @param count Denominator
     * @return The new feature
     */
    Feature operator/(size_t count) const;

    /**
     * @return X coordinate
     */
    float x() const;

    /**
     * @return Y coordinate
     */
    float y() const;

    /**
     * @return R component
     */
    float r() const;

    /**
     * @return G component
     */
    float g() const;

    /**
     * @return B component
     */
    float b() const;

private:
    float m_x = 0.f, m_y = 0.f;
    float m_r = 0.f, m_g = 0.f, m_b = 0.f;
    float m_spatialWeight = 1.f;
    float m_colorWeight = 2.f;
};

#endif //HSEG_FEATURE_H
