//
// Created by jan on 28.08.16.
//

#include <BaseProperties.h>
#include <Energy/WeightsVec.h>
#include <Inference/k-prototypes/Clusterer.h>
#include <Timer.h>
#include <helper/image_helper.h>
#include <Inference/InferenceIterator.h>
#include <Accuracy/ConfusionMatrix.h>
#include <boost/filesystem/operations.hpp>
#include <Energy/feature_weights.h>

PROPERTIES_DEFINE(Inference,
                  PROP_DEFINE(size_t, numClusters, 300)
                  PROP_DEFINE(float, pairwiseSigmaSq, 0.05f)
                  PROP_DEFINE(std::string, image, "")
                  PROP_DEFINE(std::string, groundTruth, "")
                  PROP_DEFINE(std::string, unary, "")
                  PROP_DEFINE(std::string, outDir, "")
                  GROUP_DEFINE(weights,
                               PROP_DEFINE(std::string, file, "")
                               PROP_DEFINE(float, unary, 5.f)
                               PROP_DEFINE(float, pairwise, 500)
                               PROP_DEFINE(float, feature, 1.f)
                               PROP_DEFINE(float, label, 30.f)
                               PROP_DEFINE(std::string, featureWeightFile, "")
                  )
)

int main()
{
    // Read properties
    InferenceProperties properties;
    properties.read("properties/inference.info");
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    size_t const numClasses = 21;

    WeightsVec weights(numClasses, properties.weights.unary, properties.weights.pairwise, properties.weights.feature,
                       properties.weights.label);
    if(!weights.read(properties.weights.file))
        std::cerr << "Weights not read from file, using values specified in properties file!" << std::endl;
    std::cout << "Used weights:" << std::endl;
    std::cout << weights << std::endl;

    helper::image::ColorMap cmap = helper::image::generateColorMapVOC(std::max(256ul, properties.numClusters));

    // Load images
    RGBImage rgb;
    rgb.read(properties.image);
    if (rgb.pixels() == 0)
    {
        std::cerr << "Couldn't load image " << properties.image << std::endl;
        return -1;
    }
    CieLabImage cieLab = rgb.getCieLabImg();
    RGBImage groundTruthRgb;
    groundTruthRgb.read(properties.groundTruth);
    bool useGroundTruth = true;
    if (groundTruthRgb.width() != rgb.width() || groundTruthRgb.height() != rgb.height())
    {
        std::cout << "Specified ground truth image is invalid. Won't show accuracy." << std::endl;
        useGroundTruth = false;
    }

    // Convert ground truth back to indices
    LabelImage groundTruth;
    if (useGroundTruth)
        groundTruth = helper::image::decolorize(groundTruthRgb, cmap);

    // Load unary scores
    UnaryFile unaryFile(properties.unary);
    if (!unaryFile.isValid() || unaryFile.classes() != numClasses || unaryFile.width() != rgb.width() ||
        unaryFile.height() != rgb.height())
    {
        std::cerr << "Unary file is invalid." << std::endl;
        return -2;
    }

    // Read in feature weights
    Matrix5 featureWeights = readFeatureWeights(properties.weights.featureWeightFile);
    featureWeights = featureWeights.inverse();
    std::cout << "Used feature weights:" << std::endl;
    std::cout << featureWeights << std::endl;

    // Create energy function
    EnergyFunction energyFun(unaryFile, weights, properties.pairwiseSigmaSq, featureWeights);

    // Do the inference!
    Timer t(true);
    InferenceIterator<EnergyFunction> inference(energyFun, properties.numClusters, numClasses, cieLab);
    auto result = inference.runDetailed();
    t.pause();

    std::cout << "Computed " << result.numIter << " iterations (" << t.elapsed<Timer::milliseconds>() << ")" << std::endl;
    std::cout << "Energy after each iteration: ";
    for(size_t i = 0; i < result.energy.size(); ++i)
        std::cout << i << "\t" << result.energy[i] << std::endl;

    // Compute accuracy
    if (useGroundTruth)
    {
        ConfusionMatrix unaryAccuracy(numClasses, unaryFile.maxLabeling(), groundTruth);
        std::cout << "Accuracy (unary): " << unaryAccuracy << std::endl;
        for(size_t i = 0; i < result.numIter; ++i)
        {
            ConfusionMatrix accuracy(numClasses, result.labelings[i], groundTruth);
            std::cout << "Accuracy (iteration " << std::to_string(i) << "): " << accuracy << std::endl;
        }
    }

    // Write results to disk
    std::string filename = boost::filesystem::path(properties.image).stem().string();
    boost::filesystem::path basePath(properties.outDir + "/" + filename + "/");
    boost::filesystem::remove_all(basePath);
    boost::filesystem::path spPath(basePath.string() + "/sp/");
    boost::filesystem::create_directories(spPath);
    boost::filesystem::path labelPath(basePath.string() + "/labeling/");
    boost::filesystem::create_directories(labelPath);
    for(size_t i = 0; i < result.numIter; ++i)
    {
        cv::Mat labelMat = static_cast<cv::Mat>(helper::image::colorize(result.labelings[i], cmap));
        cv::Mat spMat = static_cast<cv::Mat>(helper::image::colorize(result.superpixels[i], cmap));
        cv::imwrite(spPath.string() + std::to_string(i + 1) + ".png", spMat);
        cv::imwrite(labelPath.string() + std::to_string(i + 1) + ".png", labelMat);
    }
    LabelImage unaryLabeling = unaryFile.maxLabeling();
    cv::Mat labelMat = static_cast<cv::Mat>(helper::image::colorize(unaryLabeling, cmap));
    cv::imwrite(basePath.string() + "unary.png", labelMat);
    cv::Mat rgbMat = static_cast<cv::Mat>(rgb);
    cv::imwrite(basePath.string() + "rgb.png", rgbMat);


    // Show results
    if (useGroundTruth)
    {
        cv::Mat gtMat = static_cast<cv::Mat>(groundTruthRgb);
        cv::imshow("ground truth", gtMat);
    }
    cv::imshow("rgb", rgbMat);

    for(size_t i = 0; i < result.numIter; ++i)
    {
        cv::Mat labelMat = static_cast<cv::Mat>(helper::image::colorize(result.labelings[i], cmap));
        cv::Mat spMat = static_cast<cv::Mat>(helper::image::colorize(result.superpixels[i], cmap));
        cv::imshow("labeling (" + std::to_string(i) + ")", labelMat);
        cv::imshow("superpixels (" + std::to_string(i) + ")", spMat);
    }
    cv::waitKey();


    return 0;
}