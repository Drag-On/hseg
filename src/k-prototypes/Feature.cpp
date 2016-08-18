//
// Created by jan on 18.08.16.
//

#include <helper/coordinate_helper.h>
#include "k-prototypes/Feature.h"

Feature::Feature(RGBImage const& rgb, size_t site)
        : m_r(rgb.at(site, 0)),
          m_g(rgb.at(site, 1)),
          m_b(rgb.at(site, 2))
{ // TODO: Have weights that balance coordinates vs. color
    auto coords = helper::coord::siteTo2DCoordinate(site, rgb.width());
    m_x = coords.first;
    m_y = coords.second;
}

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
    return std::pow(m_x - other.m_x, 2) + std::pow(m_y - other.m_y, 2) + std::pow(m_r - other.m_r, 2) +
                     std::pow(m_g - other.m_g, 2) + std::pow(m_b - other.m_b, 2);
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
    m_x /= count;
    m_y /= count;
    m_r /= count;
    m_g /= count;
    m_b /= count;
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
