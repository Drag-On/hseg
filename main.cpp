#include <iostream>
#include <UnaryFile.h>
#include <cv.hpp>
#include "GCoptimization.h"

int main()
{
    std::cout << "Hello, World!" << std::endl;

    UnaryFile unary("data/2007_000129_prob.dat");

    GCoptimizationGridGraph* gc = new GCoptimizationGridGraph(unary.height(), unary.width(), unary.classes());
    for (int x = 0; x < unary.width(); ++x)
        for (int y = 0; y < unary.height(); ++y)
            for (int l = 0; l < unary.classes(); ++l)
                gc->setDataCost(x + y * unary.width(), l, -unary.at(x, y, l));
    gc->expansion();

    cv::Mat result(unary.height(), unary.width(), CV_8UC1);
    for (int x = 0; x < unary.width(); ++x)
        for (int y = 0; y < unary.height(); ++y)
            result.at<unsigned char>(y, x) = gc->whatLabel(x + y * unary.width());

    delete gc;

    /*cv::Mat result(unary.height(), unary.width(), CV_8UC1);
    for (int x = 0; x < unary.width(); ++x)
        for (int y = 0; y < unary.height(); ++y)
            result.at<unsigned char>(y, x) = unary.maxLabelAt(x, y);*/

    cv::equalizeHist( result, result );

    cv::imshow("result", result);
    cv::waitKey();

    return 0;
}