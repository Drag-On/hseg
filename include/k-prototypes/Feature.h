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
        m_x = coords.x();
        m_y = coords.y();
        m_r = color.atSite(site, 0);
        m_g = color.atSite(site, 1);
        m_b = color.atSite(site, 2);
    }

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
     * Squares all elements of this feature
     */
    void squareElements();

    /**
     * @return Feature with all elements squared
     */
    Feature getSquaredElements() const;

    /**
     * @return X coordinate
     */
    inline float x() const
    {
        return m_x;
    }

    /**
     * @return Y coordinate
     */
    inline float y() const
    {
        return m_y;
    }

    /**
     * @return R component
     */
    inline float r() const
    {
        return m_r;
    }

    /**
     * @return G component
     */
    inline float g() const
    {
        return m_g;
    }

    /**
     * @return B component
     */
    inline float b() const
    {
        return m_b;
    }

private:
    float m_x = 0.f, m_y = 0.f;
    float m_r = 0.f, m_g = 0.f, m_b = 0.f;
};

#endif //HSEG_FEATURE_H
