//
// Created by jan on 01.09.16.
//

#include <BaseProperties.h>
#include <Energy/WeightsVec.h>
#include <Energy/UnaryFile.h>
#include <Energy/LossAugmentedEnergyFunction.h>
#include <helper/image_helper.h>
#include <Inference/InferenceIterator.h>

PROPERTIES_DEFINE(TrainDistPred,
                  PROP_DEFINE(size_t, numClusters, 300)
                  PROP_DEFINE(float, pairwiseSigmaSq, 0.05f)
                  PROP_DEFINE(std::string, weightFile, "")
                  PROP_DEFINE(std::string, imageFile, "")
                  PROP_DEFINE(std::string, groundTruthFile, "")
                  PROP_DEFINE(std::string, groundTruthSpFile, "")
                  PROP_DEFINE(std::string, unaryFile, "")
                  PROP_DEFINE(std::string, out, "out/weights.dat")
)

/**
 * Arguments:
 *  1 - Image file
 *  2 - Ground truth file
 *  3 - Ground truth superpixel file
 *  4 - Unary file
 *  5 - Output file
 *  6 - Weight file
 * @param argc
 * @param argv
 * @param properties
 */
void parseArguments(int argc, char* argv[], TrainDistPredProperties& properties)
{
    if (argc > 1)
        properties.imageFile = std::string(argv[1]);
    if (argc > 2)
        properties.groundTruthFile = std::string(argv[2]);
    if (argc > 3)
        properties.groundTruthSpFile = std::string(argv[3]);
    if (argc > 4)
        properties.unaryFile = std::string(argv[4]);
    if (argc > 5)
        properties.out = std::string(argv[5]);
    if (argc > 6)
        properties.weightFile = std::string(argv[6]);
}

int main(int argc, char* argv[])
{
    // Read properties
    TrainDistPredProperties properties;
    properties.read("properties/training_dist_pred.info");
    parseArguments(argc, argv, properties);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    size_t const numClasses = 21;
    size_t const numClusters = properties.numClusters;
    helper::image::ColorMap const cmap = helper::image::generateColorMapVOC(std::max(256ul, numClasses));
    helper::image::ColorMap const cmap2 = helper::image::generateColorMap(properties.numClusters);
    WeightsVec oneWeights(numClasses, 1, 1, 1, 1, 1, 1, 1);
    WeightsVec curWeights(numClasses);
    if(!curWeights.read(properties.weightFile))
    {
        std::cerr << "Couldn't read current weights from " << properties.weightFile << std::endl;
        return -1;
    }

    // Load images etc...
    RGBImage rgbImage, groundTruthRGB, groundTruthSpRGB;
    rgbImage.read(properties.imageFile);
    groundTruthRGB.read(properties.groundTruthFile);
    groundTruthSpRGB.read(properties.groundTruthSpFile);
    if (rgbImage.width() != groundTruthRGB.width() || rgbImage.height() != groundTruthRGB.height() ||
        rgbImage.width() != groundTruthSpRGB.width() || rgbImage.height() != groundTruthSpRGB.height())
    {
        std::cerr << "Image " << properties.imageFile << " and its ground truth don't match." << std::endl;
        return -2;
    }
    CieLabImage cieLabImage = rgbImage.getCieLabImg();
    LabelImage groundTruth = helper::image::decolorize(groundTruthRGB, cmap);
    LabelImage groundTruthSp = helper::image::decolorize(groundTruthSpRGB, cmap2);

    UnaryFile unary(properties.unaryFile);
    if(unary.width() != rgbImage.width() || unary.height() != rgbImage.height() || unary.classes() != numClasses)
    {
        std::cerr << "Invalid unary scores " << properties.unaryFile << std::endl;
        return -3;
    }

    // Predict with loss-augmented energy
    LossAugmentedEnergyFunction energy(unary, curWeights, properties.pairwiseSigmaSq, groundTruth);
    InferenceIterator inference(energy, properties.numClusters, numClasses, cieLabImage);
    InferenceResult result = inference.run(2);

    // Compute energy without weights on the ground truth
    EnergyFunction normalEnergy(unary, oneWeights, properties.pairwiseSigmaSq);
    auto clusters = Clusterer::computeClusters(groundTruthSp, cieLabImage, groundTruth, properties.numClusters, numClasses);
    auto gtEnergy = normalEnergy.giveEnergyByWeight(groundTruth, cieLabImage, groundTruthSp, clusters);

    // Compute energy without weights on the prediction
    auto predEnergy = normalEnergy.giveEnergyByWeight(result.labeling, cieLabImage, result.superpixels,
                                                      result.clusterer.clusters());

    // Compute difference and store result
    gtEnergy -= predEnergy;
    if(!gtEnergy.write(properties.out))
    {
        std::cerr << "Couldn't write result to " << properties.out << std::endl;
        return -4;
    }

    return 0;
}