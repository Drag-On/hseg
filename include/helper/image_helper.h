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
         * Generates a colormap with \p n distinct entries
         * @param n Amount of entries. May not be larger than 556
         * @return The generated colormap
         */
        ColorMap generateColorMap(size_t n);

        /**
         * Colorizes a label image according to a color map
         * @param labelImg Image to colorize
         * @param colorMap Color map
         * @return The colored image
         */
        RGBImage colorize(LabelImage const& labelImg, ColorMap const& colorMap);

        /**
         * Reverts the colorization by a color map
         * @param rgb Color image that has been colorized
         * @param colorMap Color map that has been used to colorize the image
         * @return The index image
         * @note The Pascal VOC ground truth images have white pixels which are not part of the classes. Therefore, for
         *       these images, the color map should have 255 entries (the 255th is white) or have white manually added.
         */
        LabelImage decolorize(RGBImage const& rgb, ColorMap const& colorMap);
    }
}

#endif