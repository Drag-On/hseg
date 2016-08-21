//
// Created by jan on 21.08.16.
//

#ifndef HSEG_COORDINATE_H
#define HSEG_COORDINATE_H

#include <array>
#include <assert.h>

/**
 * Coordinate in some dimension
 * @tparam T Type of the coordinate
 * @tparam N Dimensionality
 */
template<typename T, unsigned int N>
class Coordinates
{
public:
    /**
     * Access operator
     * @param i Element to access
     * @return The element
     */
    T const& operator[](size_t i) const
    {
        assert(i < N);
        return m_data[i];
    }

    /**
     * Access operator
     * @param i Element to access
     * @return The element
     */
    T& operator[](size_t i)
    {
        assert(i < N);
        return m_data[i];
    }

private:
    std::array<T, N> m_data;
};

/**
 * Coordinate in 2 dimensions
 * @tparam T Type of the coordinate
 */
template<typename T>
class Coordinates<T, 2>
{
public:
    Coordinates(T x = 0, T y = 0)
            : m_data{x, y}
    {
    }

    T const& x() const
    {
        return m_data[0];
    }

    T& x()
    {
        return m_data[0];
    }

    T const& y() const
    {
        return m_data[1];
    }

    T& y()
    {
        return m_data[1];
    }

    T const& operator[](size_t i) const
    {
        assert(i < 2);
        return m_data[i];
    }

    T& operator[](size_t i)
    {
        assert(i < 2);
        return m_data[i];
    }

private:
    std::array<T, 2> m_data;
};

/**
 * Coordinate in 3 dimensions
 * @tparam T Type of the coordinate
 */
template<typename T>
class Coordinates<T, 3>
{
public:
    Coordinates(T x = 0, T y = 0, T z = 0)
            : m_data{x, y, z}
    {
    }

    T const& x() const
    {
        return m_data[0];
    }

    T& x()
    {
        return m_data[0];
    }

    T const& y() const
    {
        return m_data[1];
    }

    T& y()
    {
        return m_data[1];
    }

    T const& z() const
    {
        return m_data[2];
    }

    T& z()
    {
        return m_data[2];
    }

    T const& operator[](size_t i) const
    {
        assert(i < 3);
        return m_data[i];
    }

    T& operator[](size_t i)
    {
        assert(i < 3);
        return m_data[i];
    }

private:
    std::array<T, 3> m_data;
};

template<typename T>
using Coords2d = Coordinates<T, 2>;

template<typename T>
using Coords3d = Coordinates<T, 3>;

#endif //HSEG_COORDINATE_H
