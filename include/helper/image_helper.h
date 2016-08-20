//
// Created by jan on 18.08.16.
//

#ifndef HSEG_IMAGE_HELPER_H
#define HSEG_IMAGE_HELPER_H

#include <vector>
#include <array>
#include <Image/Image.h>

namespace helper
{
    namespace image
    {
        /**
         * Maps indices to colors
         */
        template <typename T>
        using GeneralColorMap = std::vector<std::array<T, 3>>;

        /**
         * Colormap where colors are stored as unsigned bytes
         */
        using ColorMap = GeneralColorMap<unsigned char>;

        /**
         * Generates a colormap with \p n entries with the algorithm used by PascalVOC to colorize the resulting labeling
         * @param n Amount of entries
         * @return The generated colormap
         */
        ColorMap generateColorMapVOC(size_t n);

        /**
         * Colorizes a label image according to a color map
         * @param labelImg Image to colorize
         * @param colorMap Color map
         * @return The colored image
         */
        RGBImage colorize(LabelImage const& labelImg, ColorMap const& colorMap);
    }
}

#endif