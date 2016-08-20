#include <iostream>
#include <UnaryFile.h>
#include <k-prototypes/Clusterer.h>
#include <helper/image_helper.h>
#include <GraphOptimizer/GraphOptimizer.h>
#include <Energy/EnergyFunction.h>


int main()
{
    std::string filename = "2007_000129"; //"2007_000032";

    UnaryFile unary("data/" + filename + "_prob.dat");

    RGBImage rgb;
    rgb.read("data/" + filename + ".jpg");
    CieLabImage cieLab = rgb.getCieLabImg();
    LabelImage maxLabeling = unary.maxLabeling();

    size_t numClusters = 400;

    LabelImage fakeSpLabeling(unary.width(), unary.height());
    EnergyFunction energyFun(unary);
    float lastEnergy, energy = energyFun.giveEnergy(maxLabeling, cieLab, fakeSpLabeling);
    std::cout << "Energy before anything: " << energy << std::endl;

    float eps = 1000.f;
    LabelImage spLabeling;
    LabelImage classLabeling = maxLabeling;
    Clusterer clusterer(energyFun);
    GraphOptimizer optimizer(unary);
    do
    {
        lastEnergy = energy;

        clusterer.run(numClusters, unary.classes(), cieLab, classLabeling);
        spLabeling = clusterer.clustership();

        energy = energyFun.giveEnergy(classLabeling, cieLab, spLabeling);
        std::cout << "Energy after clustering: " << energy << std::endl;

        optimizer.run(cieLab, spLabeling, numClusters);
        classLabeling = optimizer.labeling();

        energy = energyFun.giveEnergy(classLabeling, cieLab, spLabeling);
        std::cout << "Energy after labeling: " << energy << std::endl;

        std::cout << "Energy decreased by " << lastEnergy - energy << " (threshold is " << eps << ")" << std::endl;
    } while (lastEnergy - energy > eps);


    helper::image::ColorMap cmap = helper::image::generateColorMapVOC(std::max<int>(unary.classes(), numClusters));

    cv::Mat rgbMat = static_cast<cv::Mat>(rgb);
    cv::Mat labelMat = static_cast<cv::Mat>(helper::image::colorize(maxLabeling, cmap));
    cv::Mat spLabelMat = static_cast<cv::Mat>(helper::image::colorize(spLabeling, cmap));
    cv::Mat newLabelMat = static_cast<cv::Mat>(helper::image::colorize(classLabeling, cmap));

    cv::imshow("max labeling", labelMat);
    cv::imshow("rgb", rgbMat);
    cv::imshow("sp", spLabelMat);
    cv::imshow("class labeling", newLabelMat);
    cv::waitKey();

    return 0;
}