//
// Created by jan on 01.09.16.
//

#include <BaseProperties.h>
#include <Energy/WeightsVec.h>
#include <Energy/UnaryFile.h>
#include <Energy/LossAugmentedEnergyFunction.h>
#include <helper/image_helper.h>
#include <Inference/InferenceIterator.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <Energy/feature_weights.h>

PROPERTIES_DEFINE(TrainDistPred,
                  PROP_DEFINE_A(size_t, numClusters, 300, -c)
                  PROP_DEFINE_A(float, pairwiseSigmaSq, 1.00166e-06, -s)
                  PROP_DEFINE_A(std::string, weightFile, "", -w)
                  PROP_DEFINE_A(std::string, imageFile, "", -i)
                  PROP_DEFINE_A(std::string, groundTruthFile, "", -g)
                  PROP_DEFINE_A(std::string, groundTruthSpFile, "", -gsp)
                  PROP_DEFINE_A(std::string, unaryFile, "", -u)
                  PROP_DEFINE_A(std::string, featureWeightFile, "", -fw)
                  PROP_DEFINE_A(std::string, out, "out/", -o)
)

int main(int argc, char* argv[])
{
    // Read properties
    TrainDistPredProperties properties;
    properties.read("properties/training_dist_pred.info");
    properties.fromCmd(argc, argv);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    // Check if there already is a result file
    boost::filesystem::path imgNamePath(properties.imageFile);
    std::string imgName = imgNamePath.filename().stem().string();
    std::string labelPath = properties.out + "labeling/" + imgName + ".png";
    std::string spPath = properties.out + "sp/" + imgName + ".png";
    if(boost::filesystem::exists(labelPath) && boost::filesystem::exists(spPath))
    {
        std::cout << "Result for " << properties.imageFile << " already exists in " << labelPath << " and " << spPath << ". Skipping." << std::endl;
        return 0;
    }

    size_t const numClasses = 21;
    size_t const numClusters = properties.numClusters;
    helper::image::ColorMap const cmap = helper::image::generateColorMapVOC(std::max(256ul, numClasses));
    helper::image::ColorMap const cmap2 = helper::image::generateColorMap(properties.numClusters);
    WeightsVec curWeights(numClasses, 0, 0, 0, 0);
    if(!curWeights.read(properties.weightFile))
    {
        std::cout << "Couldn't read current weights from " << properties.weightFile << std::endl;
        std::cout << "Using default weights. This is only right if this is the first iteration." << std::endl;
    }

    // Load images etc...
    RGBImage rgbImage, groundTruthRGB, groundTruthSpRGB;
    if(!rgbImage.read(properties.imageFile))
    {
        std::cerr << "Couldn't read image \"" << properties.imageFile << "\"" << std::endl;
        return -1;
    }
    if(!groundTruthRGB.read(properties.groundTruthFile))
    {
        std::cerr << "Couldn't read ground truth \"" << properties.groundTruthFile << "\"" << std::endl;
        return -1;
    }
    if(!groundTruthSpRGB.read(properties.groundTruthSpFile))
    {
        std::cerr << "Couldn't read superpixel ground truth \"" << properties.groundTruthSpFile << "\"" << std::endl;
        return -1;
    }
    if (rgbImage.width() != groundTruthRGB.width() || rgbImage.height() != groundTruthRGB.height())
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

    Matrix5f featureWeights = readFeatureWeights(properties.featureWeightFile);

    // Predict with loss-augmented energy
    LossAugmentedEnergyFunction energy(unary, curWeights, properties.pairwiseSigmaSq, featureWeights, groundTruth, groundTruthSp);
    InferenceIterator inference(energy, numClusters, numClasses, cieLabImage);
    InferenceResult result = inference.run();

    // Store prediction
    cv::Mat labeling = static_cast<cv::Mat>(helper::image::colorize(result.labeling, cmap));
    cv::Mat sp = static_cast<cv::Mat>(helper::image::colorize(result.superpixels, cmap2));

    if(labeling.empty() || sp.empty())
    {
        std::cerr << "Predicted labeling and/or superpixels invalid." << std::endl;
        return -4;
    }

    if(!cv::imwrite(labelPath, labeling))
    {
        std::cerr << "Couldn't write predicted labeling to \"" << labelPath << "\"" << std::endl;
        return -5;
    }
    if(!cv::imwrite(spPath, sp))
    {
        std::cerr << "Couldn't write predicted superpixels to \"" << spPath << "\"" << std::endl;
        return -6;
    }

    return 0;
}