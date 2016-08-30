#include <iostream>
#include <Energy/UnaryFile.h>
#include <helper/image_helper.h>
#include <boost/filesystem.hpp>
#include <Energy/WeightsVec.h>
#include <map>
#include <set>
#include <Energy/EnergyFunction.h>
#include "Timer.h"

std::vector<Cluster> computeClusters(LabelImage const& sp, CieLabImage const& cieLab, LabelImage const& labeling, size_t numClusters, size_t numClasses)
{
    std::vector<Cluster> clusters(numClusters, Cluster(numClasses));
    for(size_t i = 0; i < sp.pixels(); ++i)
    {
        clusters[sp.atSite(i)].accumFeature += Feature(cieLab, i);
        ++clusters[sp.atSite(i)].size;
        ++clusters[sp.atSite(i)].labelFrequencies[labeling.atSite(i)];
    }
    for(auto& c : clusters)
    {
        c.updateMean();
        c.updateLabel();
    }
    return clusters;
}

int main()
{
    std::string filename = "2007_000129";
    std::string groundTruthFolder = "/home/jan/Downloads/Pascal VOC/data/VOC2012/SegmentationClass/";
    std::string groundTruthSpFolder = "/home/jan/Dokumente/Git/hseg/spGroundTruth/";

    size_t numClusters = 300;
    size_t numClasses = 21;
    WeightsVec weights(numClasses, 5, 500, 0.1, 0.9, 0.9, 0.9, 220);
    helper::image::ColorMap cmap = helper::image::generateColorMapVOC(std::max(256ul, numClasses));
    helper::image::ColorMap cmap2 = helper::image::generateColorMap(numClusters);
    UnaryFile unary("data/" + filename + "_prob.dat");
    LabelImage maxLabeling = unary.maxLabeling();

    /*
     * Load images
     */
    RGBImage rgb;
    std::string actualFile = "data/" + filename + ".jpg";
    rgb.read(actualFile);
    if (rgb.pixels() == 0)
    {
        std::cerr << "Couldn't load image " << actualFile << std::endl;
        return -1;
    }
    CieLabImage cieLab = rgb.getCieLabImg();
    RGBImage groundTruthRGB;
    groundTruthRGB.read(groundTruthFolder + filename + ".png");
    if (groundTruthRGB.pixels() == 0)
    {
        std::cerr << "Couldn't load ground truth image" << std::endl;
        return -2;
    }
    LabelImage groundTruth = helper::image::decolorize(groundTruthRGB, cmap);
    RGBImage groundTruthSpRGB;
    groundTruthSpRGB.read(groundTruthSpFolder + filename + ".png");
    if (groundTruthRGB.pixels() == 0)
    {
        std::cerr << "Couldn't load ground truth superpixel image" << std::endl;
        return -3;
    }
    LabelImage groundTruthSp = helper::image::decolorize(groundTruthSpRGB, cmap2);
    std::cout << std::endl << "Loaded image " << actualFile << std::endl;

    /*
     * Print some statistics
     */
    std::cout << "image size: " << cieLab.width() << "x" << cieLab.height() << " (" << rgb.width() << "x" << rgb.height() << ")" << std::endl;
    std::cout << "gt size: " << groundTruth.width() << "x" << groundTruth.height() << " (" << groundTruthRGB.width() << "x" << groundTruthRGB.height() << ")" << std::endl;
    std::cout << "gt sp size: " << groundTruthSp.width() << "x" << groundTruthSp.height() << " (" << groundTruthSpRGB.width() << "x" << groundTruthSpRGB.height() << ")" << std::endl;
    std::set<size_t> found;
    for(size_t i = 0; i < groundTruthSp.pixels(); ++i)
        found.insert(groundTruthSp.atSite(i));
    std::cout << "Ground truth superpixel segmentation has " << found.size() << " unique indices." << std::endl;

    cv::Mat spMat = static_cast<cv::Mat>(helper::image::colorize(groundTruthSp, cmap2));
    cv::imshow("sp mat colorized", spMat);
    cv::imshow("sp mat loaded", static_cast<cv::Mat>(groundTruthSpRGB));
    cv::waitKey();


    /*
     * Check some energy
     */
    EnergyFunction energy(unary, weights);
    auto clusters = computeClusters(groundTruthSp, cieLab, groundTruth, numClusters, numClasses);
    float totalEnergy = energy.giveEnergy(groundTruth, cieLab, groundTruthSp, clusters);
    WeightsVec energyByWeights = energy.giveEnergyByWeight(groundTruth, cieLab, groundTruthSp, clusters);
    std::cout << "total energy: " << totalEnergy << std::endl; //" ( " << energy.giveUnaryEnergy(groundTruth) << ", " << energy.givePairwiseEnergy(groundTruth, cieLab) << ", " << energy.giveSpEnergy(groundTruth, cieLab, groundTruthSp, clusters) << " )" << std::endl;
    std::cout << "total energy by weights: " << energyByWeights.sum() << " ( " << energyByWeights.sumUnary() << ", " << energyByWeights.sumPairwise() << ", " << energyByWeights.sumSuperpixel() << " )" << std::endl;
    std::cout << "diff : " << totalEnergy - energyByWeights.sum() << std::endl;
    std::cout << energyByWeights << std::endl;


    return 0;
}