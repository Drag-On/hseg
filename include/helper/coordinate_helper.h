//
// Created by jan on 18.08.16.
//

#ifndef HSEG_COORDINATE_HELPER_H
#define HSEG_COORDINATE_HELPER_H

#include <cstddef>
#include <utility>
#include <tuple>

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
        inline size_t coordinateToSite(int x, int y, int width)
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
        inline size_t coordinateToSite(int x, int y, int z, int width, int height)
        {
            return x + y * width + z * width * height;
        }

        /**
         * Converts a site to a 2d coordinate
         * @param site Site
         * @param width Width
         * @return 2d coordinate
         */
        inline std::pair<int, int> siteTo2DCoordinate(size_t site, int width)
        {
            std::pair<int, int> coords;
            coords.first = site % width;
            coords.second = site / width;
            return coords;
        }

        /**
         * Converts a site to a 2d coordinate
         * @param site Site
         * @param width Width
         * @return 2d coordinate
         */
        inline std::tuple<int, int, int> siteTo3DCoordinate(size_t site, int width, int height)
        {
            std::tuple<int, int, int> coords;
            std::get<0>(coords) = site % width;
            std::get<1>(coords) = (site / width) % height;
            std::get<2>(coords) = site / (width * height);
            return coords;
        }
    }
}

#endif //HSEG_COORDINATE_HELPER_H
