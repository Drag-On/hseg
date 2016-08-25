#include <iostream>
#include <Energy/UnaryFile.h>
#include <k-prototypes/Clusterer.h>
#include <helper/image_helper.h>
#include <GraphOptimizer/GraphOptimizer.h>
#include <boost/filesystem.hpp>
#include "Timer.h"


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
        std::cout << std::endl << "Loaded image " << actualFile << std::endl;
        CieLabImage cieLab = rgb.getCieLabImg();
        LabelImage maxLabeling = unary.maxLabeling();

        LabelImage fakeSpLabeling(unary.width(), unary.height());
        std::vector<Cluster> fakeClusters(1, Cluster(unary.classes()));
        EnergyFunction energyFun(unary, weights);
        // NOTE: This first energy is not really correct because it's computed assuming that the whole image is one
        // cluster and it has null-features and label 0. It would be more correct to actually compute the mean feature
        // and the dominant label, but that's computationally heavy and I don't want to do that in the moment.
        float lastEnergy, energy = energyFun.giveEnergy(maxLabeling, cieLab, fakeSpLabeling, fakeClusters);
        std::cout << "Energy before anything: " << energy << std::endl;

        float eps = properties.convergence.overall;
        float threshold;
        float energyDecrease;
        LabelImage spLabeling;
        LabelImage classLabeling = maxLabeling;
        cv::Mat spLabelMat;
        cv::Mat newLabelMat;
        Clusterer clusterer(energyFun);
        GraphOptimizer optimizer(energyFun);
        size_t iter = 0;
        Timer timer(true);
        do
        {
            iter++;
            lastEnergy = energy;

            clusterer.run(numClusters, unary.classes(), cieLab, classLabeling);
            spLabeling = clusterer.clustership();

            energy = energyFun.giveEnergy(classLabeling, cieLab, spLabeling, clusterer.clusters());

            timer.pause();

            std::cout << iter << ": Energy after clustering: " << energy << std::endl;

            timer.start();

            optimizer.run(cieLab, spLabeling, numClusters);
            classLabeling = optimizer.labeling();

            energy = energyFun.giveEnergy(classLabeling, cieLab, spLabeling, clusterer.clusters());

            timer.pause();

            std::cout << iter << ": Energy after labeling: " << energy << std::endl;

            timer.start();

            threshold = eps * std::abs(energy);
            energyDecrease = lastEnergy - energy;

            timer.pause();

            std::cout << iter << ": Energy decreased by " << energyDecrease << " (threshold is " << threshold << ")"
                      << std::endl;

            // Write out the current labeling and segmentation
            spLabelMat = static_cast<cv::Mat>(helper::image::colorize(spLabeling, cmap));
            newLabelMat = static_cast<cv::Mat>(helper::image::colorize(classLabeling, cmap));
            boost::filesystem::path spPath("out/" + filename + "/sp/");
            boost::filesystem::create_directories(spPath);
            boost::filesystem::path labelPath("out/" + filename + "/labeling/");
            boost::filesystem::create_directories(labelPath);
            cv::imwrite(spPath.string() + std::to_string(iter) + ".png", spLabelMat);
            cv::imwrite(labelPath.string() + std::to_string(iter) + ".png", newLabelMat);

            timer.start();
        } while (energyDecrease > threshold);

        std::cout << "Time: " << timer.elapsed<Timer::seconds>() << std::endl;

        cv::Mat rgbMat = static_cast<cv::Mat>(rgb);
        cv::Mat labelMat = static_cast<cv::Mat>(helper::image::colorize(maxLabeling, cmap));

        cv::imwrite("out/" + filename + "/rgb.png", rgbMat);
        cv::imwrite("out/" + filename + "/unary.png", labelMat);

        cv::imshow("max labeling", labelMat);
        cv::imshow("rgb", rgbMat);
        cv::imshow("sp", spLabelMat);
        cv::imshow("class labeling", newLabelMat);
        cv::waitKey();
    }

    return 0;
}