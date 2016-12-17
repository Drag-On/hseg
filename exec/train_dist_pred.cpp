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
                  PROP_DEFINE_A(Label, numClusters, 300, -c)
                  PROP_DEFINE_A(float, pairwiseSigmaSq, 1.00166e-06, -s)
                  PROP_DEFINE_A(std::string, weightFile, "", -w)
                  PROP_DEFINE_A(std::string, imageFile, "", -i)
                  PROP_DEFINE_A(std::string, groundTruthFile, "", -g)
                  PROP_DEFINE_A(std::string, unaryFile, "", -u)
                  PROP_DEFINE_A(std::string, featureWeightFile, "", -fw)
                  PROP_DEFINE_A(std::string, out, "out/", -o)
                  PROP_DEFINE_A(std::string, propertiesFile, "properties/training_dist_pred.info", -p)
)

enum ErrorCode
{
    SUCCESS = 0,
    CANT_READ_IMAGE = 1,
    CANT_READ_GT = 2,
    IMAGE_GT_DONT_MATCH = 3,
    INVALID_UNARY = 4,
    INVALID_FEATURE_WEIGHTS = 5,
    INVALID_PRED_LABELING = 6,
    INVALID_PRED_SP = 7,
    INVALID_GT_SP = 8,
    CANT_WRITE_PRED_LABELING = 9,
    CANT_WRITE_PRED_SP = 10,
    CANT_WRITE_GT_SP = 11,
};

int main(int argc, char* argv[])
{
    // Read properties
    TrainDistPredProperties properties;
    properties.fromCmd(argc, argv);
    properties.read(properties.propertiesFile);
    properties.fromCmd(argc, argv); // This is so the property file location can be read via command line, however,
                                    // the command line arguments should overwrite anything written in the file,
                                    // therefore read it in again.
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    // Check if there already is a result file
    boost::filesystem::path imgNamePath(properties.imageFile);
    std::string imgName = imgNamePath.filename().stem().string();
    std::string labelPath = properties.out + "labeling/" + imgName + ".png";
    std::string spPath = properties.out + "sp/" + imgName + ".png";
    std::string bestSpPath = properties.out + "sp_gt/" + imgName + ".png";
    if(boost::filesystem::exists(labelPath) && boost::filesystem::exists(spPath) && boost::filesystem::exists(bestSpPath))
    {
        std::cout << "Result for " << properties.imageFile << " already exists in " << labelPath << " and " << spPath << ". Skipping." << std::endl;
        return SUCCESS;
    }

    Label const numClasses = 21;
    Label const numClusters = properties.numClusters;
    helper::image::ColorMap const cmap = helper::image::generateColorMapVOC(std::max<Label>(256, numClasses));
    helper::image::ColorMap const cmap2 = helper::image::generateColorMap(properties.numClusters);
    WeightsVec curWeights(numClasses, 0, 0, 1, 0);
    if(!curWeights.read(properties.weightFile))
    {
        std::cout << "Couldn't read current weights from " << properties.weightFile << std::endl;
        std::cout << "Using default weights. This is only right if this is the first iteration." << std::endl;
    }

    // Load images etc...
    RGBImage rgbImage, groundTruthRGB;
    if(!rgbImage.read(properties.imageFile))
    {
        std::cerr << "Couldn't read image \"" << properties.imageFile << "\"" << std::endl;
        return CANT_READ_IMAGE;
    }
    if(!groundTruthRGB.read(properties.groundTruthFile))
    {
        std::cerr << "Couldn't read ground truth \"" << properties.groundTruthFile << "\"" << std::endl;
        return CANT_READ_GT;
    }
    if (rgbImage.width() != groundTruthRGB.width() || rgbImage.height() != groundTruthRGB.height())
    {
        std::cerr << "Image " << properties.imageFile << " and its ground truth don't match." << std::endl;
        return IMAGE_GT_DONT_MATCH;
    }
    CieLabImage cieLabImage = rgbImage.getCieLabImg();
    LabelImage groundTruth = helper::image::decolorize(groundTruthRGB, cmap);

    UnaryFile unary(properties.unaryFile);
    if(unary.width() != rgbImage.width() || unary.height() != rgbImage.height() || unary.classes() != numClasses)
    {
        std::cerr << "Invalid unary scores " << properties.unaryFile << std::endl;
        return INVALID_UNARY;
    }

    Matrix5 featureWeights = readFeatureWeights(properties.featureWeightFile);
    featureWeights = featureWeights.inverse();
    if(featureWeights.isIdentity())
    {
        std::cerr << "Couldn't read feature weights " << properties.featureWeightFile << "!" << std::endl;
        return INVALID_FEATURE_WEIGHTS;
    }

    // Predict superpixels that best explain the ground truth
    EnergyFunction trainingEnergy(unary, curWeights, properties.pairwiseSigmaSq, featureWeights);
    Clusterer<EnergyFunction> clusterer(trainingEnergy);
    clusterer.run(numClusters, numClasses, cieLabImage, groundTruth);
    LabelImage const& bestSp = clusterer.clustership();

    // Predict with loss-augmented energy
    LossAugmentedEnergyFunction energy(unary, curWeights, properties.pairwiseSigmaSq, featureWeights, groundTruth);
    InferenceIterator<LossAugmentedEnergyFunction> inference(energy, numClusters, numClasses, cieLabImage);
    InferenceResult result = inference.run();

    // Store predictions
    cv::Mat labeling = static_cast<cv::Mat>(helper::image::colorize(result.labeling, cmap));
    cv::Mat sp = static_cast<cv::Mat>(helper::image::colorize(result.superpixels, cmap2));
    cv::Mat bestSpMat = static_cast<cv::Mat>(helper::image::colorize(bestSp, cmap2));

    if(labeling.empty())
    {
        std::cerr << "Predicted labeling invalid." << std::endl;
        return INVALID_PRED_LABELING;
    }
    if(sp.empty())
    {
        std::cerr << "Predicted superpixels invalid." << std::endl;
        return INVALID_PRED_SP;
    }
    if(bestSpMat.empty())
    {
        std::cerr << "Predicted gt superpixels invalid." << std::endl;
        return INVALID_GT_SP;
    }

    if(!cv::imwrite(labelPath, labeling))
    {
        std::cerr << "Couldn't write predicted labeling to \"" << labelPath << "\"" << std::endl;
        return CANT_WRITE_PRED_LABELING;
    }
    if(!cv::imwrite(spPath, sp))
    {
        std::cerr << "Couldn't write predicted superpixels to \"" << spPath << "\"" << std::endl;
        return CANT_WRITE_PRED_SP;
    }
    if(!cv::imwrite(bestSpPath, bestSpMat))
    {
        std::cerr << "Couldn't write predicted ground truth superpixels to \"" << bestSpPath << "\"" << std::endl;
        return CANT_WRITE_GT_SP;
    }

    return SUCCESS;
}