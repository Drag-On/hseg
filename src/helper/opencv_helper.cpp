//
// Created by jan on 19.08.16.
//

#include <opencv2/core/cvdef.h>
#include "helper/opencv_helper.h"

namespace helper
{
    namespace opencv
    {
        std::string getOpenCvTypeString(int type)
        {
            std::string typeStr;

            uchar depth = static_cast<uchar>(type & CV_MAT_DEPTH_MASK);
            uchar chans = static_cast<uchar>(1 + (type >> CV_CN_SHIFT));

            switch (depth)
            {
                case CV_8U:
                    typeStr = "8U";
                    break;
                case CV_8S:
                    typeStr = "8S";
                    break;
                case CV_16U:
                    typeStr = "16U";
                    break;
                case CV_16S:
                    typeStr = "16S";
                    break;
                case CV_32S:
                    typeStr = "32S";
                    break;
                case CV_32F:
                    typeStr = "32F";
                    break;
                case CV_64F:
                    typeStr = "64F";
                    break;
                default:
                    typeStr = "User";
                    break;
            }

            typeStr += "C";
            typeStr += (chans + '0');

            return typeStr;
        }
    }
}