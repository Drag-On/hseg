//
// Created by jan on 18.08.16.
//

#include "k-prototypes/Feature.h"

Feature& Feature::operator+=(Feature const& other)
{
    m_x += other.m_x;
    m_y += other.m_y;
    m_r += other.m_r;
    m_g += other.m_g;
    m_b += other.m_b;
    return *this;
}

Feature Feature::operator+(Feature const& other) const
{
    Feature f = *this;
    f += other;
    return f;
}

Feature& Feature::operator/=(size_t count)
{
    float const fCount = static_cast<float>(count);
    m_x /= fCount;
    m_y /= fCount;
    m_r /= fCount;
    m_g /= fCount;
    m_b /= fCount;
    return *this;
}

Feature Feature::operator/(size_t count) const
{
    Feature f = *this;
    f /= count;
    return f;
}

Feature& Feature::operator-=(Feature const& other)
{
    m_x -= other.m_x;
    m_y -= other.m_y;
    m_r -= other.m_r;
    m_g -= other.m_g;
    m_b -= other.m_b;
    return *this;
}

Feature Feature::operator-(Feature const& other)
{
    Feature f = *this;
    f -= other;
    return f;
}

Feature Feature::operator-() const
{
    Feature f = *this;
    f -= Feature();
    return f;
}

void Feature::squareElements()
{
    m_x *= m_x;
    m_y *= m_y;
    m_r *= m_r;
    m_g *= m_g;
    m_b *= m_b;
}

Feature Feature::getSquaredElements() const
{
    Feature f = *this;
    f.squareElements();
    return f;
}
