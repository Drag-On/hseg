//
// Created by jan on 18.08.16.
//

#ifndef HSEG_COORDINATE_HELPER_H
#define HSEG_COORDINATE_HELPER_H

#include <cstddef>
#include <utility>
#include <tuple>
#include "Image/Coordinates.h"

namespace helper
{
    namespace coord
    {
        /**
         * Converts a 2d coordinate to a site
         * @param x X coordinate
         * @param y Y coordinate
         * @param width Width
         * @return The site
         */
        inline size_t coordinateToSite(size_t x, size_t y, size_t width)
        {
            return x + y * width;
        }

        /**
         * Converts a 3d coordinate to a site
         * @param x X coordinate
         * @param y Y coordinate
         * @param z Z coordinate
         * @param width Width
         * @param height Height
         * @return The site
         */
        inline size_t coordinateToSite(size_t x, size_t y, size_t z, size_t width, size_t height)
        {
            return x + y * width + z * width * height;
        }

        /**
         * Converts a site to a 2d coordinate
         * @param site Site
         * @param width Width
         * @return 2d coordinate
         */
        inline Coords2d<size_t> siteTo2DCoordinate(size_t site, size_t width)
        {
            Coords2d<size_t> coords;
            coords.x() = site % width;
            coords.y() = site / width;
            return coords;
        }

        /**
         * Converts a site to a 2d coordinate
         * @param site Site
         * @param width Width
         * @return 2d coordinate
         */
        inline Coords3d<size_t> siteTo3DCoordinate(size_t site, size_t width, size_t height)
        {
            Coords3d<size_t> coords;
            coords[0] = site % width;
            coords[1] = (site / width) % height;
            coords[2] = site / (width * height);
            return coords;
        }
    }
}

#endif //HSEG_COORDINATE_HELPER_H
