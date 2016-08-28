#include <iostream>
#include <Energy/UnaryFile.h>
#include <Inference/k-prototypes/Clusterer.h>
#include <Inference/GraphOptimizer/GraphOptimizer.h>
#include <helper/image_helper.h>
#include <boost/filesystem.hpp>
#include <Accuracy/ConfusionMatrix.h>
#include "Timer.h"

int main()
{
    std::vector<std::string> files = {"2007_000027", "2007_000032", "2007_000033", "2007_000039", "2007_000042",
                                      "2007_000061", "2007_000063", "2007_000068", "2007_000121", "2007_000123",
                                      "2007_000129", "2007_000170"};
    std::string filename = "2007_000129";
    std::string groundTruthFolder = "/home/jan/Downloads/Pascal VOC/data/VOC2012/SegmentationClass/";

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
        RGBImage groundTruthRGB;
        groundTruthRGB.read(groundTruthFolder + filename + ".png");
        if(groundTruthRGB.pixels() == 0)
        {
            std::cerr << "Couldn't load ground truth image" << std::endl;
            return -2;
        }
        LabelImage groundTruth = helper::image::decolorize(groundTruthRGB, cmap);
        std::cout << std::endl << "Loaded image " << actualFile << std::endl;
        CieLabImage cieLab = rgb.getCieLabImg();
        LabelImage maxLabeling = unary.maxLabeling();

        LabelImage fakeSpLabeling(unary.width(), unary.height());
        std::vector<Cluster> fakeClusters(1, Cluster(unary.classes()));
        EnergyFunction energyFun(unary, weights);
        // NOTE: This first energy is not really correct because it's computed assuming that the whole image is one
        // cluster and it has null-features and label 0. It would be more correct to actually compute the mean feature
        // and the dominant label, but that's computationally heavy and I don't want to do that in the moment.
        float startEnergy = energyFun.giveEnergy(maxLabeling, cieLab, fakeSpLabeling, fakeClusters);

        auto diff = maxLabeling.diff(groundTruth);
        std::cout << "Unary difference to ground truth: " << diff << "/" << maxLabeling.pixels() << " or " << (float)diff/maxLabeling.pixels() << "%" << std::endl;

        ConfusionMatrix cf(unary.classes(), maxLabeling, groundTruth);
        float mean;
        auto accuracies = cf.accuracies(&mean);
        std::string str;
        properties::toString(accuracies, str);
        std::cout << "IoU accuracies: " << str << std::endl;
        std::cout << "IoU mean: " << mean << std::endl;

        int const tries = 1;
        std::vector<LabelImage> classLabelTries;
        float eps = properties.convergence.overall;
        for (int i = 0; i < tries; ++i)
        {
            std::cout << "Try " << i << std::endl;

            float lastEnergy, energy = startEnergy;
            std::cout << "Energy before anything: " << energy << std::endl;

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
                boost::filesystem::path basePath("out/" + filename + "/" + std::to_string(i) + "/");
                boost::filesystem::path spPath(basePath.string() + "/sp/");
                boost::filesystem::create_directories(spPath);
                boost::filesystem::path labelPath(basePath.string() + "/labeling/");
                boost::filesystem::create_directories(labelPath);
                cv::imwrite(spPath.string() + std::to_string(iter) + ".png", spLabelMat);
                cv::imwrite(labelPath.string() + std::to_string(iter) + ".png", newLabelMat);

                size_t diff = classLabeling.diff(groundTruth);
                std::cout << iter << ": Difference to ground truth: " << diff << "/" << classLabeling.pixels() << " or " << (float)diff/classLabeling.pixels() << "%" << std::endl;
                ConfusionMatrix cf(unary.classes(), classLabeling, groundTruth);
                float mean;
                auto accuracies = cf.accuracies(&mean);
                std::string str;
                properties::toString(accuracies, str);
                std::cout << "IoU accuracies: " << str << std::endl;
                std::cout << "IoU mean: " << mean << std::endl;

                timer.start();
            } while (energyDecrease > threshold);

            std::cout << "Time: " << timer.elapsed<Timer::seconds>() << std::endl;

            classLabelTries.push_back(classLabeling);
        }

        // Merge the tries to one final segmentation
        LabelImage finalLabeling(maxLabeling.width(), maxLabeling.height());
        for(size_t i = 0; i < finalLabeling.pixels(); ++i)
        {
            std::vector<int> classes(unary.classes(), 0);
            for(auto const& t : classLabelTries)
                classes[t.atSite(i)]++;
            Label l = std::distance(classes.begin(), std::max_element(classes.begin(), classes.end()));
            finalLabeling.atSite(i) = l;
        }

        std::cout << "Difference to ground truth: " << finalLabeling.diff(groundTruth) << std::endl;

        cv::Mat rgbMat = static_cast<cv::Mat>(rgb);
        cv::Mat labelMat = static_cast<cv::Mat>(helper::image::colorize(maxLabeling, cmap));
        cv::Mat finalLabelingMat = static_cast<cv::Mat>(helper::image::colorize(finalLabeling, cmap));

        cv::imwrite("out/" + filename + "/rgb.png", rgbMat);
        cv::imwrite("out/" + filename + "/unary.png", labelMat);
        cv::imwrite("out/" + filename + "/merged.png", finalLabelingMat);

        /*cv::imshow("max labeling", labelMat);
        cv::imshow("rgb", rgbMat);
        cv::imshow("sp", spLabelMat);
        cv::imshow("class labeling", newLabelMat);
        cv::waitKey();*/
    }

    return 0;
}