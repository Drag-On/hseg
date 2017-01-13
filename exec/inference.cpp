//
// Created by jan on 28.08.16.
//

#include <BaseProperties.h>
#include <Energy/Weights.h>
#include <Timer.h>
#include <helper/image_helper.h>
#include <Inference/InferenceIterator.h>
#include <Accuracy/ConfusionMatrix.h>
#include <boost/filesystem/operations.hpp>

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

enum ERRCODE
{
    SUCCESS=0,
    CANT_READ_FILE,
};

int main(int argc, char** argv)
{
    // Read properties
    InferenceProperties properties;
    properties.read("properties/inference.info");
    properties.fromCmd(argc, argv);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    size_t const numClasses = 21;
    size_t const featDim = 512;

    Weights weights(numClasses, featDim);
    weights.randomize();
//    if(!weights.read(properties.weights.file))
//        std::cerr << "Weights not read from file, using values specified in properties file!" << std::endl;
    std::cout << "Used weights:" << std::endl;
    std::cout << weights << std::endl;

    helper::image::ColorMap cmap = helper::image::generateColorMapVOC(256ul);

    // Load image
    FeatureImage features;
    if(!features.read(properties.image))
    {
        std::cerr << "Unable to read features from \"" << properties.image << "\"" << std::endl;
        return CANT_READ_FILE;
    }

    // Create energy function
    EnergyFunction energyFun(&weights);

    // Do the inference!
    Timer t(true);
    InferenceIterator<EnergyFunction> inference(&energyFun, &features);
    auto result = inference.runDetailed();
    t.pause();

    std::cout << "Computed " << result.numIter << " iterations (" << t.elapsed<Timer::milliseconds>() << ")" << std::endl;
    std::cout << "Energy after each iteration: " << std::endl;
    for(size_t i = 0; i < result.energy.size(); ++i)
        std::cout << i << "\t" << result.energy[i] << std::endl;

    // Compute accuracy
//    if (useGroundTruth)
//    {
//        ConfusionMatrix unaryAccuracy(numClasses, unaryFile.maxLabeling(), groundTruth);
//        std::cout << "Accuracy (unary): " << unaryAccuracy << std::endl;
//        for(size_t i = 0; i < result.numIter; ++i)
//        {
//            ConfusionMatrix accuracy(numClasses, result.labelings[i], groundTruth);
//            std::cout << "Accuracy (iteration " << std::to_string(i) << "): " << accuracy << std::endl;
//        }
//    }

    // Write results to disk
//    std::string filename = boost::filesystem::path(properties.image).stem().string();
//    boost::filesystem::path basePath(properties.outDir + "/" + filename + "/");
//    boost::filesystem::remove_all(basePath);
//    boost::filesystem::path spPath(basePath.string() + "/sp/");
//    boost::filesystem::create_directories(spPath);
//    boost::filesystem::path labelPath(basePath.string() + "/labeling/");
//    boost::filesystem::create_directories(labelPath);
//    for(size_t i = 0; i < result.numIter; ++i)
//    {
//        cv::Mat labelMat = static_cast<cv::Mat>(helper::image::colorize(result.labelings[i], cmap));
//        cv::Mat spMat = static_cast<cv::Mat>(helper::image::colorize(result.superpixels[i], cmap));
//        cv::imwrite(spPath.string() + std::to_string(i + 1) + ".png", spMat);
//        cv::imwrite(labelPath.string() + std::to_string(i + 1) + ".png", labelMat);
//    }
//    LabelImage unaryLabeling = unaryFile.maxLabeling();
//    cv::Mat labelMat = static_cast<cv::Mat>(helper::image::colorize(unaryLabeling, cmap));
//    cv::imwrite(basePath.string() + "unary.png", labelMat);
//    cv::Mat rgbMat = static_cast<cv::Mat>(rgb);
//    cv::imwrite(basePath.string() + "rgb.png", rgbMat);


    // Show results
//    if (useGroundTruth)
//    {
//        cv::Mat gtMat = static_cast<cv::Mat>(groundTruthRgb);
//        cv::imshow("ground truth", gtMat);
//    }
//    cv::imshow("rgb", rgbMat);

    for(size_t i = 0; i < result.numIter; ++i)
    {
        cv::Mat labelMat = static_cast<cv::Mat>(helper::image::colorize(result.labelings[i], cmap));
//        cv::Mat spMat = static_cast<cv::Mat>(helper::image::colorize(result.superpixels[i], cmap));
        cv::imshow("labeling (" + std::to_string(i) + ")", labelMat);
//        cv::imshow("superpixels (" + std::to_string(i) + ")", spMat);
    }
    cv::waitKey();


    return SUCCESS;
}