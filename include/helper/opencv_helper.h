//
// Created by jan on 19.08.16.
//

#ifndef HSEG_OPENCV_HELPER_H
#define HSEG_OPENCV_HELPER_H

#include <cassert>

namespace helper
{
    namespace opencv
    {
        template<typename T>
        int getOpenCvType(int channels)
        {
            assert(false);
        }

        template<>
        int getOpenCvType<unsigned char>(int channels);

        template<>
        int getOpenCvType<signed char>(int channels);

        template<>
        int getOpenCvType<unsigned short>(int channels);

        template<>
        int getOpenCvType<signed short>(int channels);

        template<>
        int getOpenCvType<signed int>(int channels);

        template<>
        int getOpenCvType<float>(int channels);

        template<>
        int getOpenCvType<double>(int channels);
    }
}

#endif //HSEG_OPENCV_HELPER_H
