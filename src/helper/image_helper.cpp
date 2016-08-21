//
// Created by jan on 18.08.16.
//

#include "helper/image_helper.h"

namespace helper
{
    namespace image
    {
        ColorMap generateColorMapVOC(size_t n)
        {
            enum Bits
            {
                FirstBit = 1 << 0,
                SecondBit = 1 << 1,
                ThirdBit = 1 << 2,
            };
            ColorMap cmap(n);
            for (size_t i = 0; i < n; ++i)
            {
                size_t id = i;
                unsigned char r = 0, g = 0, b = 0;
                for (int j = 0; j <= 7; ++j)
                {
                    // Note: This is switched compared to the pascal voc example code because opencv has BGR instead of RGB
                    b = b | static_cast<unsigned char>(((id & FirstBit) >> 0) << (7 - j));
                    g = g | static_cast<unsigned char>(((id & SecondBit) >> 1) << (7 - j));
                    r = r | static_cast<unsigned char>(((id & ThirdBit) >> 2) << (7 - j));
                    id = id >> 3;
                }
                cmap[i][0] = r;
                cmap[i][1] = g;
                cmap[i][2] = b;
            }
            return cmap;
        }

        RGBImage colorize(LabelImage const& labelImg, ColorMap const& colorMap)
        {
            RGBImage rgb(labelImg.width(), labelImg.height());

            for (size_t i = 0; i < labelImg.pixels(); ++i)
            {
                Label l = labelImg.atSite(i);
                assert(colorMap.size() > l);

                rgb.atSite(i, 0) = colorMap[l][0];
                rgb.atSite(i, 1) = colorMap[l][1];
                rgb.atSite(i, 2) = colorMap[l][2];
            }

            return rgb;
        }
    }
}
