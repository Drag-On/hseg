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

        /**
         * Draws outlines onto a color image according to a label image
         * @param labelImg Label image
         * @param colorImg Color image
         * @return Color image with overlayed outlines
         */
        RGBImage outline(LabelImage const& labelImg, RGBImage const& colorImg, std::array<unsigned short, 3> const& color = {255, 255, 255});

        enum class PNGError
        {
            Okay = 0,
            CantOpenFile,
            InvalidFileFormat,
            CantInitReadStruct,
            CantInitWriteStruct,
            CantInitInfoStruct,
            Critical,
            NoPalette,
            UnsupportedInterlaceType,
            OutOfMemory,
        };

        /**
         * Reads in a png file which is stored in a palette format
         * @param file Filename
         * @param outImage Label image to write the index image to
         * @param pOutColorMap The color map is stored here. Set to nullptr to ignore.
         * @return Error code
         */
        PNGError readPalettePNG(std::string const& file, LabelImage& outImage, ColorMap* pOutColorMap);

        /**
         * Writes a label image to png with a color palette
         * @param file Filename
         * @param labeling Label image
         * @param cmap Color map to use
         * @return Error code
         */
        PNGError writePalettePNG(std::string const& file, LabelImage const& labeling, ColorMap const& cmap);

        /**
         * Provides a MAP prediction in a given size by using the marginals of a different size
         * @param img Color image
         * @param marginals Marginals
         * @returns MAP prediction of the size of \p img
         */
        LabelImage
        rescaleMAP(RGBImage const& img, std::vector<Image<double, 1>> const& marginals);

        /**
         * Reads a marginal distribution from file
         * @param file File to read from
         * @param outMarginals Marginals will be stored here
         * @return True in case of success, otherwise false
         */
        bool readMarginals(std::string const& file, std::vector<Image<double, 1>>& outMarginals);

        /**
         * Writes a marginal distribution to file
         * @param file File to write to
         * @param marginals Marginals to write
         * @return True in case of success, otherwise false
         */
        bool writeMarginals(std::string const& file, std::vector<Image<double, 1>> const& marginals);
    }
}

#endif