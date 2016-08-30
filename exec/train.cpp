//
// Created by jan on 29.08.16.
//

#include <BaseProperties.h>
#include <Energy/WeightsVec.h>
#include <Energy/UnaryFile.h>
#include <Energy/LossAugmentedEnergyFunction.h>
#include <helper/image_helper.h>
#include <Inference/InferenceIterator.h>
#include <Inference/k-prototypes/Clusterer.h>

PROPERTIES_DEFINE(Train,
                  PROP_DEFINE(size_t, numClusters, 300)
                  PROP_DEFINE(size_t, numIter, 100)
                  PROP_DEFINE(float, learningRate, 1.f)
                  PROP_DEFINE(float, C, 1.f)
                  PROP_DEFINE(float, pairwiseSigmaSq, 0.05f)
                  PROP_DEFINE(std::string, imageListFile, "")
                  PROP_DEFINE(std::string, groundTruthListFile, "")
                  PROP_DEFINE(std::string, groundTruthSpListFile, "")
                  PROP_DEFINE(std::string, unaryListFile, "")
                  PROP_DEFINE(std::string, imageBasePath, "")
                  PROP_DEFINE(std::string, groundTruthBasePath, "")
                  PROP_DEFINE(std::string, groundTruthSpBasePath, "")
                  PROP_DEFINE(std::string, unaryBasePath, "")
                  PROP_DEFINE(std::string, imageExtension, ".jpg")
                  PROP_DEFINE(std::string, gtExtension, ".png")
                  PROP_DEFINE(std::string, outDir, "out/")
)

std::vector<std::string> readFileNames(std::string const& listFile)
{
    std::vector<std::string> list;
    std::ifstream in(listFile, std::ios::in);
    if (in.is_open())
    {
        std::string line;
        while (std::getline(in, line))
            list.push_back(line);
        in.close();
    }
    return list;
}

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
    // Read properties
    TrainProperties properties;
    properties.read("properties/training.info");
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    size_t const numClasses = 21;
    helper::image::ColorMap const cmap = helper::image::generateColorMapVOC(std::max(256ul, numClasses));
    helper::image::ColorMap const cmap2 = helper::image::generateColorMap(properties.numClusters);
    WeightsVec curWeights(numClasses, false); // Zero-weights
    WeightsVec oneWeights(numClasses, 1, 1, 1, 1, 1, 1, 1);

    // Load filenames of all images
    std::vector<std::string> colorImageFilenames = readFileNames(properties.imageListFile);
    std::vector<std::string> gtImageFilenames = readFileNames(properties.groundTruthListFile);
    std::vector<std::string> gtSpImageFilenames = readFileNames(properties.groundTruthSpListFile);
    std::vector<std::string> unaryFilenames = readFileNames(properties.unaryListFile);
    if (colorImageFilenames.size() != gtImageFilenames.size() || gtImageFilenames.size() != unaryFilenames.size() ||
        gtImageFilenames.size() != gtSpImageFilenames.size())
    {
        std::cerr << "File lists don't match up!" << std::endl;
        return -1;
    }
    size_t T = properties.numIter;
    size_t N = colorImageFilenames.size();

    // Iterate T times
    for(size_t t = 0; t < T; ++t)
    {
        // Iterate over all images
        for(size_t n = 0; n < N; ++n)
        {
            // Load images etc...
            RGBImage rgbImage, groundTruthRGB, groundTruthSpRGB;
            rgbImage.read(properties.imageBasePath + colorImageFilenames[n] + properties.imageExtension);
            groundTruthRGB.read(properties.groundTruthBasePath + gtImageFilenames[n] + properties.gtExtension);
            groundTruthSpRGB.read(properties.groundTruthSpBasePath + gtSpImageFilenames[n] + properties.gtExtension);
            if (rgbImage.width() != groundTruthRGB.width() || rgbImage.height() != groundTruthRGB.height() ||
                rgbImage.width() != groundTruthSpRGB.width() || rgbImage.height() != groundTruthSpRGB.height())
            {
                std::cerr << "Image " << colorImageFilenames[n] << " and its ground truth don't match." << std::endl;
                continue;
            }
            CieLabImage cieLabImage = rgbImage.getCieLabImg();
            LabelImage groundTruth = helper::image::decolorize(groundTruthRGB, cmap);

            //cv::imshow("sp mat loaded", static_cast<cv::Mat>(groundTruthSpRGB));
            //cv::waitKey();

            LabelImage groundTruthSp = helper::image::decolorize(groundTruthSpRGB, cmap2);
            UnaryFile unary(properties.unaryBasePath + unaryFilenames[n] + "_prob.dat");
            if(unary.width() != rgbImage.width() || unary.height() != rgbImage.height() || unary.classes() != numClasses)
            {
                std::cerr << "Invalid unary scores " << unaryFilenames[n] << std::endl;
                continue;
            }

            // Predict with loss-augmented energy
            LossAugmentedEnergyFunction energy(unary, curWeights, properties.pairwiseSigmaSq, groundTruth);
            InferenceIterator inference(energy, properties.numClusters, numClasses, cieLabImage);
            InferenceResult result = inference.run(2);

            // Compute energy without weights on the ground truth
            EnergyFunction normalEnergy(unary, oneWeights, properties.pairwiseSigmaSq);
            auto clusters = computeClusters(groundTruthSp, cieLabImage, groundTruth, properties.numClusters, numClasses);
            auto gtEnergy = normalEnergy.giveEnergyByWeight(groundTruth, cieLabImage, groundTruthSp, clusters);

            // Compute energy without weights on the prediction
            clusters = computeClusters(result.superpixels, cieLabImage, result.labeling, properties.numClusters, numClasses);
            auto predEnergy = normalEnergy.giveEnergyByWeight(result.labeling, cieLabImage, result.superpixels,
                                                              clusters);

            //float byWeight = predEnergy.sum();
            //float normal = normalEnergy.giveEnergy(result.labeling, cieLabImage, result.superpixels, clusters);
            //assert(std::abs(byWeight - normal) < std::max(byWeight, normal) * 0.001f);

            std::cout << "<<< " << t << "/" << n << " >>>" << std::endl;

            // Update step
            gtEnergy -= predEnergy;
            //std::cout << "-----------------" << std::endl;
            //std::cout << gtEnergy << std::endl;
            gtEnergy *= properties.C / N;
            //std::cout << "-----------------" << std::endl;
            //std::cout << gtEnergy << std::endl;
            gtEnergy += curWeights;
            //std::cout << "-----------------" << std::endl;
            //std::cout << gtEnergy << std::endl;
            gtEnergy *= properties.learningRate / (t + n + 1);
            //std::cout << "-----------------" << std::endl;
            //std::cout << gtEnergy << std::endl;
            curWeights -= gtEnergy;
            //std::cout << "-----------------" << std::endl;
            std::cout << curWeights << std::endl;
        }
    }

    return 0;
}