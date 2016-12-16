//
// Created by jan on 18.08.16.
//

#ifndef HSEG_COORDINATE_HELPER_H
#define HSEG_COORDINATE_HELPER_H

#include <cstddef>
#include <utility>
#include <tuple>
#include "Image/Coordinates.h"
#include "typedefs.h"

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
        inline SiteId coordinateToSite(Coord x, Coord y, Coord width)
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
        inline SiteId coordinateToSite(Coord x, Coord y, Coord z, Coord width, Coord height)
        {
            return x + y * width + z * width * height;
        }

        /**
         * Converts a site to a 2d coordinate
         * @param site Site
         * @param width Width
         * @return 2d coordinate
         */
        inline Coords2d<Coord> siteTo2DCoordinate(SiteId site, Coord width)
        {
            Coords2d<Coord> coords;
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
        inline Coords3d<Coord> siteTo3DCoordinate(SiteId site, Coord width, Coord height)
        {
            Coords3d<Coord> coords;
            coords[0] = site % width;
            coords[1] = (site / width) % height;
            coords[2] = site / (width * height);
            return coords;
        }
    }
}

#endif //HSEG_COORDINATE_HELPER_H
