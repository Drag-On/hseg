#include <iostream>
#include <UnaryFile.h>
#include <k-prototypes/Clusterer.h>
#include <helper/image_helper.h>
#include <GraphOptimizer/GraphOptimizer.h>
#include <Energy/EnergyFunction.h>
#include <Properties.h>


int main()
{
    std::string filename = "2007_000129"; //"2007_000032";

    UnaryFile unary("data/" + filename + "_prob.dat");

    RGBImage rgb;
    rgb.read("data/" + filename + ".jpg");
    CieLabImage cieLab = rgb.getCieLabImg();
    LabelImage maxLabeling = unary.maxLabeling();

    HsegProperties properties;
    properties.read("properties.info");

    size_t numClusters = properties.clustering.numClusters;

    LabelImage fakeSpLabeling(unary.width(), unary.height());
    EnergyFunction energyFun(unary, properties.weights);
    float lastEnergy, energy = energyFun.giveEnergy(maxLabeling, cieLab, fakeSpLabeling);
    std::cout << "Energy before anything: " << energy << std::endl;

    float eps = properties.convergence.overall;
    float threshold;
    float energyDecrease;
    LabelImage spLabeling;
    LabelImage classLabeling = maxLabeling;
    Clusterer clusterer(energyFun);
    GraphOptimizer optimizer(energyFun);
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

        threshold = eps * std::abs(energy);
        energyDecrease = lastEnergy - energy;
        std::cout << "Energy decreased by " << energyDecrease << " (threshold is " << threshold << ")" << std::endl;
    } while (energyDecrease > threshold);


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