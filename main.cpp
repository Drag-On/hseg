#include <iostream>
#include <Energy/UnaryFile.h>
#include <k-prototypes/Clusterer.h>
#include <helper/image_helper.h>
#include <GraphOptimizer/GraphOptimizer.h>


int main()
{
    std::vector<std::string> files = {"2007_000027", "2007_000032", "2007_000033", "2007_000039", "2007_000042",
                                      "2007_000061", "2007_000063", "2007_000068", "2007_000121", "2007_000123",
                                      "2007_000129", "2007_000170"};
    std::string filename = "2007_000129";

    HsegProperties properties;
    properties.read("properties.info");
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    Weights weights(21ul);

    size_t numClusters = properties.clustering.numClusters;
    helper::image::ColorMap cmap = helper::image::generateColorMapVOC(std::max(21ul, numClusters));

    //for (std::string filename : files)
    {
        UnaryFile unary("data/" + filename + "_prob.dat");

        RGBImage rgb;
        std::string actualFile = "data/" + filename + ".jpg";
        rgb.read(actualFile);
        if (rgb.pixels() == 0)
        {
            std::cerr << "Couldn't load image " << actualFile << std::endl;
            return -1;
        }
        std::cout  << std::endl << "Loaded image " << actualFile << std::endl;
        CieLabImage cieLab = rgb.getCieLabImg();
        LabelImage maxLabeling = unary.maxLabeling();

        LabelImage fakeSpLabeling(unary.width(), unary.height());
        EnergyFunction energyFun(unary, weights);
        float lastEnergy, energy = energyFun.giveEnergy(maxLabeling, cieLab, fakeSpLabeling);
        std::cout << "Energy before anything: " << energy << std::endl;

        float eps = properties.convergence.overall;
        float threshold;
        float energyDecrease;
        LabelImage spLabeling;
        LabelImage classLabeling = maxLabeling;
        Clusterer clusterer(energyFun);
        GraphOptimizer optimizer(energyFun);
        size_t iter = 0;
        do
        {
            iter++;
            lastEnergy = energy;

            clusterer.run(numClusters, unary.classes(), cieLab, classLabeling);
            spLabeling = clusterer.clustership();

            energy = energyFun.giveEnergy(classLabeling, cieLab, spLabeling);
            std::cout << iter << ": Energy after clustering: " << energy << std::endl;

            optimizer.run(cieLab, spLabeling, numClusters);
            classLabeling = optimizer.labeling();

            energy = energyFun.giveEnergy(classLabeling, cieLab, spLabeling);
            std::cout << iter << ": Energy after labeling: " << energy << std::endl;

            threshold = eps * std::abs(energy);
            energyDecrease = lastEnergy - energy;
            std::cout << iter << ": Energy decreased by " << energyDecrease << " (threshold is " << threshold << ")" << std::endl;
        } while (energyDecrease > threshold);

        cv::Mat rgbMat = static_cast<cv::Mat>(rgb);
        cv::Mat labelMat = static_cast<cv::Mat>(helper::image::colorize(maxLabeling, cmap));
        cv::Mat spLabelMat = static_cast<cv::Mat>(helper::image::colorize(spLabeling, cmap));
        cv::Mat newLabelMat = static_cast<cv::Mat>(helper::image::colorize(classLabeling, cmap));

        cv::imshow("max labeling", labelMat);
        cv::imshow("rgb", rgbMat);
        cv::imshow("sp", spLabelMat);
        cv::imshow("class labeling", newLabelMat);
        cv::waitKey();

        /*cv::imwrite("out/" + filename + "_unary.png", labelMat);
        cv::imwrite("out/" + filename + "_rgb.png", rgbMat);
        cv::imwrite("out/" + filename + "_sp.png", spLabelMat);
        cv::imwrite("out/" + filename + "_labeling.png", newLabelMat);*/
    }

    return 0;
}