//
// Created by jan on 28.08.16.
//

#include <BaseProperties.h>
#include <Energy/WeightsVec.h>
#include <Inference/k-prototypes/Clusterer.h>
#include <helper/image_helper.h>
#include <Inference/InferenceIterator.h>
#include <boost/filesystem/operations.hpp>
#include <Threading/ThreadPool.h>

PROPERTIES_DEFINE(InferenceBatch,
                  PROP_DEFINE(size_t, numClusters, 300)
                  PROP_DEFINE(float, pairwiseSigmaSq, 0.05f)
                  PROP_DEFINE(std::string, imageList, "")
                  PROP_DEFINE(std::string, imageDir, "")
                  PROP_DEFINE(std::string, imageExtension, ".jpg")
                  PROP_DEFINE(std::string, unaryDir, "")
                  PROP_DEFINE(std::string, unaryExtension, ".dat")
                  PROP_DEFINE(std::string, outDir, "")
                  PROP_DEFINE(unsigned short, numThreads, 4)
                  GROUP_DEFINE(weights,
                               PROP_DEFINE(std::string, file, "")
                               PROP_DEFINE(float, unary, 5.f)
                               PROP_DEFINE(float, pairwise, 500)
                               GROUP_DEFINE(feature,
                                            PROP_DEFINE(float, a, 0.1f)
                                            PROP_DEFINE(float, b, 0.9f)
                                            PROP_DEFINE(float, c, 0.9f)
                                            PROP_DEFINE(float, d, 0.0f)
                               )
                               PROP_DEFINE(float, label, 30.f)
                  )
)

bool process(std::string const& imageFilename, std::string const& unaryFilename, size_t classes, size_t clusters,
             WeightsVec const& weights, helper::image::ColorMap const& map, InferenceBatchProperties const& properties)
{
    // Load images
    RGBImage rgb;
    rgb.read(imageFilename);
    if (rgb.pixels() == 0)
    {
        std::cerr << "Couldn't load image " << imageFilename << std::endl;
        return false;
    }
    CieLabImage cieLab = rgb.getCieLabImg();

    // Load unary scores
    UnaryFile unaryFile(unaryFilename);
    if (!unaryFile.isValid() || unaryFile.classes() != classes || unaryFile.width() != rgb.width() ||
        unaryFile.height() != rgb.height())
    {
        std::cerr << "Unary file is invalid." << std::endl;
        return false;
    }

    // Create energy function
    EnergyFunction energyFun(unaryFile, weights, properties.pairwiseSigmaSq);

    // Do the inference!
    InferenceIterator inference(energyFun, clusters, classes, cieLab);
    auto result = inference.run(2);

    // Write results to disk
    std::string filename = boost::filesystem::path(imageFilename).stem().string();
    boost::filesystem::path spPath(properties.outDir + "/sp/");
    boost::filesystem::create_directories(spPath);
    boost::filesystem::path labelPath(properties.outDir + "/labeling/");
    boost::filesystem::create_directories(labelPath);
    cv::Mat labelMat = static_cast<cv::Mat>(helper::image::colorize(result.labeling, map));
    cv::Mat spMat = static_cast<cv::Mat>(helper::image::colorize(result.superpixels, map));
    cv::imwrite(spPath.string() + filename + ".png", spMat);
    cv::imwrite(labelPath.string() + filename + ".png", labelMat);

    return true;
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

int main()
{
    // Read properties
    InferenceBatchProperties properties;
    properties.read("properties/inference_batch.info");
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    size_t const numClasses = 21;
    size_t const numClusters = properties.numClusters;

    WeightsVec weights(numClasses, properties.weights.unary, properties.weights.pairwise, properties.weights.feature.a,
                       properties.weights.feature.b, properties.weights.feature.c, properties.weights.feature.d,
                       properties.weights.label);
    if(!weights.read(properties.weights.file))
        std::cerr << "Weights not read from file, using values specified in properties file!" << std::endl;
    std::cout << "Used weights:" << std::endl;
    std::cout << weights << std::endl;

    helper::image::ColorMap const cmap = helper::image::generateColorMapVOC(std::max(256ul, properties.numClusters));

    // Read in file names to process
    auto filenames = readFileNames(properties.imageList);
    if(filenames.empty())
    {
        std::cerr << "No files specified." << std::endl;
        return -1;
    }

    // Clear output directory
    boost::filesystem::path basePath(properties.outDir);
    boost::filesystem::remove_all(basePath);

    ThreadPool pool(properties.numThreads);
    std::vector<std::future<bool>> futures;

    // Iterate all files
    for(auto const& f : filenames)
    {
        std::string const& imageFilename = properties.imageDir + f + properties.imageExtension;
        std::string const& unaryFilename = properties.unaryDir + f + properties.unaryExtension;
        auto&& fut = pool.enqueue(process, imageFilename, unaryFilename, numClasses, numClusters, weights, cmap, properties);
        futures.push_back(std::move(fut));
    }

    // Wait for all the threads to finish
    for(size_t i = 0; i < futures.size(); ++i)
    {
        bool ok = futures[i].get();
        if(!ok)
            std::cerr << "Couldn't process image \"" + filenames[i] + "\"" << std::endl;
        else
            std::cout << "Done with \"" + filenames[i] + "\"" << std::endl;
    }

    return 0;
}