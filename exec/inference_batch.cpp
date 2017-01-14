//
// Created by jan on 28.08.16.
//

#include <BaseProperties.h>
#include <Energy/Weights.h>
#include <helper/image_helper.h>
#include <Inference/InferenceIterator.h>
#include <boost/filesystem/operations.hpp>
#include <Threading/ThreadPool.h>

PROPERTIES_DEFINE(InferenceBatch,
                  PROP_DEFINE_A(size_t, numClusters, 300, -c)
                  PROP_DEFINE(float, pairwiseSigmaSq, 1.00166e-06)
                  PROP_DEFINE_A(std::string, imageList, "", -s)
                  PROP_DEFINE(std::string, imageDir, "")
                  PROP_DEFINE(std::string, imageExtension, ".jpg")
                  PROP_DEFINE(std::string, unaryDir, "")
                  PROP_DEFINE(std::string, unaryExtension, ".dat")
                  PROP_DEFINE(std::string, outDir, "")
                  PROP_DEFINE(unsigned short, numThreads, 4)
                  GROUP_DEFINE(weights,
                               PROP_DEFINE_A(std::string, file, "", -w)
                               PROP_DEFINE(float, unary, 5.f)
                               PROP_DEFINE(float, pairwise, 500)
                               PROP_DEFINE(float, feature, 1)
                               PROP_DEFINE(float, label, 30.f)
                               PROP_DEFINE(std::string, featureWeightFile, "")
                  )
)

enum EXIT_CODE
{
    SUCCESS = 0,
    FILE_LIST_EMPTY,

};

struct Result
{
    bool okay = false;
    std::string filename;
};

Result process(std::string const& imageFilename, std::string const& unaryFilename, size_t classes, size_t clusters,
             Weights const& weights, helper::image::ColorMap const& map, InferenceBatchProperties const& properties,
             Matrix5 const& featureWeights, std::string const& spOutPath, std::string const& labelOutPath)
{
    std::string filename = boost::filesystem::path(imageFilename).stem().string();
    Result res;
    res.filename = filename;

    // Load images
    RGBImage rgb;
    rgb.read(imageFilename);
    if (rgb.pixels() == 0)
    {
        std::cerr << "Couldn't load image " << imageFilename << std::endl;
        return res;
    }
    CieLabImage cieLab = rgb.getCieLabImg();

    // Load unary scores
    UnaryFile unaryFile(unaryFilename);
    if (!unaryFile.isValid() || unaryFile.classes() != classes || unaryFile.width() != rgb.width() ||
        unaryFile.height() != rgb.height())
    {
        std::cerr << "Unary file is invalid." << std::endl;
        return res;
    }

    // Create energy function
    EnergyFunction energyFun(unaryFile, weights, properties.pairwiseSigmaSq, featureWeights);

    // Do the inference!
    InferenceIterator<EnergyFunction> inference(energyFun, clusters, classes, cieLab);
    auto result = inference.run();

    // Write results to disk
    boost::filesystem::path spPath(spOutPath);
    boost::filesystem::create_directories(spPath);
    boost::filesystem::path labelPath(labelOutPath);
    boost::filesystem::create_directories(labelPath);
    cv::Mat labelMat = static_cast<cv::Mat>(helper::image::colorize(result.labeling, map));
    cv::Mat spMat = static_cast<cv::Mat>(helper::image::colorize(result.superpixels, map));
    cv::imwrite(spPath.string() + filename + ".png", spMat);
    cv::imwrite(labelPath.string() + filename + ".png", labelMat);

    res.okay = true;
    return res;
}

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

int main(int argc, char** argv)
{
    // Read properties
    InferenceBatchProperties properties;
    properties.read("properties/inference_batch.info");
    properties.fromCmd(argc, argv);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    size_t const numClasses = 21;
    size_t const numClusters = properties.numClusters;

    Weights weights(numClasses, properties.weights.unary, properties.weights.pairwise, properties.weights.feature,
                       properties.weights.label);
    if(!weights.read(properties.weights.file))
        std::cerr << "Weights not read from file, using values specified in properties file!" << std::endl;
    std::cout << "Used weights:" << std::endl;
    std::cout << weights << std::endl;

    // Load feature weights
    Matrix5 featureWeights = readFeatureWeights(properties.weights.featureWeightFile);
    featureWeights = featureWeights.inverse();
    std::cout << "Used feature weights:" << std::endl;
    std::cout << featureWeights << std::endl;

    helper::image::ColorMap const cmap = helper::image::generateColorMapVOC(256ul);

    // Read in file names to process
    auto filenames = readFileNames(properties.imageList);
    if(filenames.empty())
    {
        std::cerr << "No files specified." << std::endl;
        return FILE_LIST_EMPTY;
    }

    boost::filesystem::path spPath(properties.outDir + "/sp/");
    boost::filesystem::path labelPath(properties.outDir + "/labeling/");

    // Clear output directory
    boost::filesystem::path basePath(properties.outDir);
    std::cout << "Clear output directories in " << basePath << "? (y/N) ";
    std::string response;
    std::getline(std::cin, response);
    if (response == "y" || response == "Y")
    {
        boost::filesystem::remove_all(spPath);
        boost::filesystem::remove_all(labelPath);
    }

    ThreadPool pool(properties.numThreads);
    std::vector<std::future<Result>> futures;

    // Iterate all files
    for(auto const& f : filenames)
    {
        std::string const& imageFilename = properties.imageDir + f + properties.imageExtension;
        std::string const& unaryFilename = properties.unaryDir + f + properties.unaryExtension;
        std::string filename = boost::filesystem::path(imageFilename).stem().string();
        if(boost::filesystem::exists(spPath / (filename + ".png")) && boost::filesystem::exists(labelPath / (filename + ".png")))
        {
            std::cout << "Skipping " << f << "." << std::endl;
            continue;
        }
        auto&& fut = pool.enqueue(process, imageFilename, unaryFilename, numClasses, numClusters, weights, cmap, properties, featureWeights, spPath.string(), labelPath.string());
        futures.push_back(std::move(fut));
    }

    // Wait for all the threads to finish
    for(size_t i = 0; i < futures.size(); ++i)
    {
        Result res = futures[i].get();
        if(!res.okay)
            std::cerr << "Couldn't process image \"" + res.filename + "\"" << std::endl;
        else
            std::cout << "Done with \"" + res.filename + "\"" << std::endl;
    }

    return SUCCESS;
}