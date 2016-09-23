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
    for (size_t i = 0; i < sp.pixels(); ++i)
    {
        assert(sp.atSite(i) < numClusters);
        clusters[sp.atSite(i)].accumFeature += Feature(cieLab, i);
        clusters[sp.atSite(i)].size++;
        if (labeling.atSite(i) < numClasses)
            clusters[sp.atSite(i)].labelFrequencies[labeling.atSite(i)]++;
    }
    for (auto& c : clusters)
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
    std::string weightsFile = "/home/jan/Dokumente/Git/hseg/out/weights.dat";

    size_t numClusters = 300;
    size_t numClasses = 21;
    WeightsVec weights(numClasses, 5, 500, 0.1, 0.9, 0.9, 0.9, 220);
    weights.read(weightsFile);
    std::cout << weights << std::endl;
    helper::image::ColorMap cmap = helper::image::generateColorMapVOC(std::max(256ul, numClasses));
    helper::image::ColorMap cmap2 = helper::image::generateColorMap(numClusters);
    UnaryFile unary("data/" + filename + "_prob.dat");
    LabelImage maxLabeling = unary.maxLabeling();

    weights.unaryWeights() = { 1077.29, 1434.4, 1088.44, 1583.51, 1360.1, 1747.57, 1712.61, 1632.54, 1993.21, 1073.46, 1587.16, 1254.3, 1960.55, 1561.02, 1380.94, 881.735, 1142.99, 1584.13, 1448.46, 1766.74, 1613.3 };
    weights.pairwiseWeights() = { 19.7293, 9.92478, 0.00577019, 16.5797, 0.0303489, 0.00209402, 13.5655, 0.098511, 0.0115467, 0.0966735, 9.53504, 0.00123235, 0.00625781, 0.0106053, 0.00497726, 31.7596, 0.0804732, 0.0553034, 0.000743949, 0.0538299, 0.00686675, 25.1488, 0.383724, 0.141541, 0.25131, 0.363245, 0.0479459, 0.899997, 61.0322, 0.0211205, 0.067921, 0.229577, 0.00230963, 0.0732108, 0.416343, 0.175153, 22.5943, 0.0162991, 0.129974, 0.0195839, 0.0783281, 0.0933275, 0.0515332, 0.166302, 1.67329, 13.0211, 0.00715809, 0.0034381, 0.0270357, 0.0967292, 0.000484931, 0.050813, 0.140632, 0.0680422, 0.0161927, 12.109, 0.303794, 0.0685025, 0.648379, 0.292451, 0.99499, 0.20527, 0.665673, 1.7298, 4.86023, 0.131481, 31.9275, 0.000384504, 0.00786174, 0.0335846, 0.0137218, 0.00397016, 0.0404198, 0.184389, 1.71992, 0.433977, 0.0107753, 0.302725, 15.313, 0.0107938, 0.0243197, 0.0118573, 0.0214658, 0.00148668, 0.417423, 0.197675, 0.772001, 0.0813662, 0.183706, 0.0883543, 0.145842, 18.1027, 0.0246888, 0.122964, 0.00101556, 0.0315655, 0.00287644, 0.535904, 0.335541, 1.15248, 0.0479904, 0.0398566, 0.372699, 0.232674, 0.53797, 82.9551, 1.01879, 1.16206, 0.540772, 1.46942, 1.98711, 1.65977, 3.42718, 2.03639, 3.86607, 0.638292, 5.00463, 2.28479, 3.29078, 5.359, 10.6724, 0.0136841, 0.0480336, 0.223802, 0.0646271, 0.0751263, 0.0371587, 0.349, 0.566902, 0.381758, 0.0578435, 0.533557, 0.0572712, 0.0583538, 0.0520894, 0.956519, 14.5321, 0.0247554, 0.0659891, 0.109241, 0.0221035, -0.00210418, 0.350412, 0.178745, 2.91917, 0.0522097, 0.244642, 0.395202, 0.433909, 1.50805, 0.964983, 1.16676, 0.0746819, 18.5782, 0.0211751, 0.0967715, 0.0483735, 0.0213626, 0.0269412, 0.173985, 0.192077, 2.81202, 1.88535, 0.0286724, 1.04721, 1.70345, 0.109162, 0.180159, 5.7358, 0.191058, 0.315619, 26.7174, 0.0770211, 0.0167546, 0.0280285, 0.160896, 0.0282329, 0.31171, 0.371714, 0.0850279, 0.150666, 0.021963, 0.198597, 0.0035669, 0.0825313, 0.162625, 0.86528, 0.157754, 0.121354, 0.0770748, 17.6388, 0.0896317, 0.0241463, 0.176011, 0.0836033, 0.359742, 0.0584805, 0.416975, 0.423116, 0.656661, 0.0296478, 0.41084, 0.0632638, 0.0499363, 0.00522647, 1.75341, 0.492544, 0.0481771, 0.537536, 0.307669, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
    weights.featureWeights() = { 1.64441e+06, 3.11994e+07, 2.31301e+07, 35670.8 };
    weights.classWeight() = 8543.33;
    weights.write(weightsFile);

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