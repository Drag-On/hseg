//
// Created by jan on 19.08.16.
//

#ifndef HSEG_OPENCV_HELPER_H
#define HSEG_OPENCV_HELPER_H

#include <cassert>
#include <string>
#include <boost/type_index.hpp>

namespace helper
{
    namespace opencv
    {
        template<typename T>
        int getOpenCvType(int channels)
        {
            if (typeid(T) == typeid(unsigned char))
                return CV_8UC(channels);
            else if (typeid(T) == typeid(signed char))
                return CV_8SC(channels);
            else if (typeid(T) == typeid(unsigned short))
                return CV_16UC(channels);
            else if (typeid(T) == typeid(signed short))
                return CV_16SC(channels);
            else if (typeid(T) == typeid(signed int))
                return CV_32SC(channels);
            else if (typeid(T) == typeid(float))
                return CV_32FC(channels);
            else if (typeid(T) == typeid(double))
                return CV_64FC(channels);
            else
                throw boost::typeindex::type_id<T>().pretty_name() + " is not a valid openCV type.";
        }
    }

    std::string getOpenCvTypeString(int type);
}

#endif //HSEG_OPENCV_HELPER_H
