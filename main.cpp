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

    LabelImage fakeSpLabeling(unary.width(), unary.height());
    EnergyFunction energyFun(unary);
    float energy = energyFun.giveEnergy(maxLabeling, cieLab, fakeSpLabeling);
    std::cout << "Energy before anything: " << energy << std::endl;

    size_t numClusters = 120;
    Clusterer clusterer(energyFun);
    clusterer.run(numClusters, unary.classes(), cieLab, maxLabeling);
    LabelImage const& spLabeling = clusterer.clustership();

    energy = energyFun.giveEnergy(maxLabeling, cieLab, spLabeling);
    std::cout << "Energy after clustering: " << energy << std::endl;

    GraphOptimizer optimizer(unary);
    optimizer.run(cieLab, spLabeling, numClusters);
    LabelImage const& newLabeling = optimizer.labeling();

    energy = energyFun.giveEnergy(newLabeling, cieLab, spLabeling);
    std::cout << "Energy after labeling: " << energy << std::endl;

    helper::image::ColorMap cmap = helper::image::generateColorMapVOC(std::max<int>(unary.classes(), numClusters));

    cv::Mat rgbMat = static_cast<cv::Mat>(rgb);
    cv::Mat labelMat = static_cast<cv::Mat>(helper::image::colorize(maxLabeling, cmap));
    cv::Mat spLabelMat = static_cast<cv::Mat>(helper::image::colorize(spLabeling, cmap));
    cv::Mat newLabelMat = static_cast<cv::Mat>(helper::image::colorize(newLabeling, cmap));

    cv::imshow("max labeling", labelMat);
    cv::imshow("rgb", rgbMat);
    cv::imshow("sp", spLabelMat);
    cv::imshow("new labeling", newLabelMat);
    cv::waitKey();

    return 0;
}