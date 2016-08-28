//
// Created by jan on 28.08.16.
//

#include <BaseProperties.h>
#include <Energy/Weights.h>
#include <Inference/k-prototypes/Clusterer.h>
#include <Timer.h>
#include <helper/image_helper.h>
#include <Inference/InferenceIterator.h>
#include <Accuracy/ConfusionMatrix.h>
#include <boost/filesystem/operations.hpp>

PROPERTIES_DEFINE(Inference,
                  PROP_DEFINE(size_t, numClusters, 300)
                  PROP_DEFINE(std::string, image, "")
                  PROP_DEFINE(std::string, groundTruth, "")
                  PROP_DEFINE(std::string, unary, "")
                  PROP_DEFINE(std::string, outDir, "")
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

    Weights weights(numClasses); // TODO: Load weights instead of always using the default ones
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

    // Create energy function
    EnergyFunction energyFun(unaryFile, weights);

    // Do the inference!
    Timer t(true);
    InferenceIterator inference(energyFun, properties.numClusters, numClasses, cieLab);
    auto result = inference.runDetailed();
    t.pause();

    std::cout << "Computed " << result.numIter << " iterations (" << t.elapsed<Timer::milliseconds>() << ")" << std::endl;
    std::cout << "Energy after each iteration: ";
    for(size_t i = 0; i < result.energy.size() - 1; ++i)
        std::cout << result.energy[i] << ", ";
    std::cout << result.energy.back() << std::endl;

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
        cv::imwrite(spPath.string() + std::to_string(i) + ".png", spMat);
        cv::imwrite(labelPath.string() + std::to_string(i) + ".png", labelMat);
    }

    // Show results
    if (useGroundTruth)
    {
        cv::Mat gtMat = static_cast<cv::Mat>(groundTruthRgb);
        cv::imshow("ground truth", gtMat);
    }
    cv::Mat rgbMat = static_cast<cv::Mat>(rgb);
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