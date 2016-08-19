//
// Created by jan on 19.08.16.
//

#include <opencv2/core/cvdef.h>
#include "helper/opencv_helper.h"

namespace helper
{
    namespace opencv
    {
        template<>
        int getOpenCvType<unsigned char>(int channels)
        {
            return CV_8UC(channels);
        }

        template<>
        int getOpenCvType<signed char>(int channels)
        {
            return CV_8SC(channels);
        }

        template<>
        int getOpenCvType<unsigned short>(int channels)
        {
            return CV_16UC(channels);
        }

        template<>
        int getOpenCvType<signed short>(int channels)
        {
            return CV_16SC(channels);
        }

        template<>
        int getOpenCvType<signed int>(int channels)
        {
            return CV_32SC(channels);
        }

        template<>
        int getOpenCvType<float>(int channels)
        {
            return CV_32FC(channels);
        }

        template<>
        int getOpenCvType<double>(int channels)
        {
            return CV_64FC(channels);
        }
    }
}