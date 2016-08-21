//
// Created by jan on 18.08.16.
//

#include "k-prototypes/Feature.h"

float Feature::x() const
{
    return m_x;
}

float Feature::y() const
{
    return m_y;
}

float Feature::r() const
{
    return m_r;
}

float Feature::g() const
{
    return m_g;
}

float Feature::b() const
{
    return m_b;
}

float Feature::sqDistanceTo(Feature const& other) const
{
    auto xDiff = m_x - other.m_x;
    auto yDiff = m_y - other.m_y;
    auto rDiff = m_r - other.m_r;
    auto gDiff = m_g - other.m_g;
    auto bDiff = m_b - other.m_b;
    return xDiff * xDiff + yDiff * yDiff + rDiff * rDiff + gDiff * gDiff + bDiff * bDiff;
}

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
    float fCount = static_cast<float>(count);
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
