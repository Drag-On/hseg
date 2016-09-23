//
// Created by jan on 29.08.16.
//

#include <BaseProperties.h>
#include <Energy/WeightsVec.h>
#include <Energy/UnaryFile.h>
#include <Energy/LossAugmentedEnergyFunction.h>
#include <helper/image_helper.h>
#include <Inference/InferenceIterator.h>

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
                  PROP_DEFINE(std::string, out, "out/weights.dat")
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

struct SampleResult
{
    WeightsVec energyDiff{21ul, false};
    float trainingEnergy = 0;
    bool valid = false;
};

SampleResult processSample(std::string const& colorImgFilename, std::string const& gtImageFilename,
                           std::string const& gtSpImageFilename, std::string const& unaryFilename,
                           TrainProperties const& properties, helper::image::ColorMap const& cmap,
                           helper::image::ColorMap const& cmap2, size_t numClasses, WeightsVec const& curWeights,
                           WeightsVec const& oneWeights)
{
    SampleResult sampleResult;

    // Load images etc...
    RGBImage rgbImage, groundTruthRGB, groundTruthSpRGB;
    rgbImage.read(properties.imageBasePath + colorImgFilename + properties.imageExtension);
    groundTruthRGB.read(properties.groundTruthBasePath + gtImageFilename + properties.gtExtension);
    groundTruthSpRGB.read(properties.groundTruthSpBasePath + gtSpImageFilename + properties.gtExtension);
    if (rgbImage.width() != groundTruthRGB.width() || rgbImage.height() != groundTruthRGB.height() ||
        rgbImage.width() != groundTruthSpRGB.width() || rgbImage.height() != groundTruthSpRGB.height())
    {
        std::cerr << "Image " << colorImgFilename << " and its ground truth don't match." << std::endl;
        return sampleResult;
    }
    CieLabImage cieLabImage = rgbImage.getCieLabImg();
    LabelImage groundTruth = helper::image::decolorize(groundTruthRGB, cmap);
    LabelImage groundTruthSp = helper::image::decolorize(groundTruthSpRGB, cmap2);

    UnaryFile unary(properties.unaryBasePath + unaryFilename + "_prob.dat");
    if(unary.width() != rgbImage.width() || unary.height() != rgbImage.height() || unary.classes() != numClasses)
    {
        std::cerr << "Invalid unary scores " << unaryFilename << std::endl;
        return sampleResult;
    }

    // Predict with loss-augmented energy
    LossAugmentedEnergyFunction energy(unary, curWeights, properties.pairwiseSigmaSq, groundTruth);
    InferenceIterator inference(energy, properties.numClusters, numClasses, cieLabImage);
    InferenceResult result = inference.run(2);

    sampleResult.trainingEnergy = energy.giveEnergy(result.labeling, cieLabImage, result.superpixels, result.clusterer.clusters());

    // Compute energy without weights on the ground truth
    EnergyFunction normalEnergy(unary, oneWeights, properties.pairwiseSigmaSq);
    auto clusters = Clusterer::computeClusters(groundTruthSp, cieLabImage, groundTruth, properties.numClusters, numClasses);
    auto gtEnergy = normalEnergy.giveEnergyByWeight(groundTruth, cieLabImage, groundTruthSp, clusters);

    // Compute energy without weights on the prediction
    auto predEnergy = normalEnergy.giveEnergyByWeight(result.labeling, cieLabImage, result.superpixels,
                                                      result.clusterer.clusters());
    // Compute energy difference
    gtEnergy -= predEnergy;

    sampleResult.energyDiff = gtEnergy;
    sampleResult.valid = true;
    return sampleResult;
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
    WeightsVec curWeights(numClasses, 1, 0, 0, 0, 0, 0, 0); // Start with the result from the unary only
    WeightsVec oneWeights(numClasses, 1, 1, 1, 1, 1, 1, 1);

    std::cout << "====================" << std::endl;
    std::cout << "Initial weights:" << std::endl;
    std::cout << curWeights << std::endl;

    // Load filenames of all images
    std::vector<std::string> colorImageFilenames = readFileNames(properties.imageListFile);
    std::vector<std::string> gtImageFilenames = readFileNames(properties.groundTruthListFile);
    std::vector<std::string> gtSpImageFilenames = readFileNames(properties.groundTruthSpListFile);
    std::vector<std::string> unaryFilenames = readFileNames(properties.unaryListFile);
    if (colorImageFilenames.size() != gtImageFilenames.size() || gtImageFilenames.size() != unaryFilenames.size() ||
        gtImageFilenames.size() != gtSpImageFilenames.size())
    {
        std::cerr << "File lists don't match up!" << std::endl;
        return 1;
    }
    size_t T = properties.numIter;
    size_t N = colorImageFilenames.size();

    // Iterate T times
    for(size_t t = 0; t < T; ++t)
    {
        WeightsVec sum(numClasses, 0, 0, 0, 0, 0, 0, 0); // All zeros
        float iterationEnergy = 0;

        // Iterate over all images
        for (size_t n = 0; n < N; ++n)
        {
            auto colorImgFilename = colorImageFilenames[n];
            auto gtImageFilename = gtImageFilenames[n];
            auto gtSpImageFilename = gtSpImageFilenames[n];
            auto unaryFilename = unaryFilenames[n];

            auto sampleResult = processSample(colorImgFilename, gtImageFilename, gtSpImageFilename, unaryFilename,
                                              properties, cmap, cmap2, numClasses, curWeights, oneWeights);

            if(!sampleResult.valid)
            {
                std::cerr << "Sample result was invalid. Cannot continue." << std::endl;
                return 2;
            }

            std::cout << "<<< " << t << "/" << n << " >>>" << std::endl;

            sum += sampleResult.energyDiff;
            iterationEnergy += sampleResult.trainingEnergy;
        }

        // Show current training energy
        iterationEnergy *= properties.C / N;
        iterationEnergy += 1.f / 2.f * curWeights.sqNorm();
        std::cout << "Current training energy: " << iterationEnergy << std::endl;

        // Update step
        sum *= properties.C / N;
        sum += curWeights;
        sum *= properties.learningRate / (t + 1);
        curWeights -= sum;

        if (!curWeights.write(properties.out))
            std::cerr << "Couldn't write weights to file " << properties.out << std::endl;

        std::cout << "====================" << std::endl;
        std::cout << "Current weights:" << std::endl;
        std::cout << curWeights << std::endl;
    }

    return 0;
}