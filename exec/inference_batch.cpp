//
// Created by jan on 28.08.16.
//

#include <BaseProperties.h>
#include <Energy/Weights.h>
#include <helper/image_helper.h>
#include <helper/clustering_helper.h>
#include <Inference/InferenceIterator.h>
#include <boost/filesystem/operations.hpp>
#include <Threading/ThreadPool.h>

PROPERTIES_DEFINE(InferenceBatch,
                  GROUP_DEFINE(dataset,
                               PROP_DEFINE_A(std::string, list, "", -l)
                               GROUP_DEFINE(path,
                                            PROP_DEFINE_A(std::string, img, "", --img)
                                            PROP_DEFINE_A(std::string, gt, "", --gt)
                               )
                               GROUP_DEFINE(extension,
                                            PROP_DEFINE_A(std::string, img, ".mat", --img_ext)
                                            PROP_DEFINE_A(std::string, gt, ".png", --gt_ext)
                               )
                               GROUP_DEFINE(constants,
                                            PROP_DEFINE_A(uint32_t, numClasses, 21, --numClasses)
                                            PROP_DEFINE_A(uint32_t, featDim, 512, --featDim)
                               )
                  )
                  GROUP_DEFINE(param,
                          PROP_DEFINE_A(std::string, weights, "", -w)
                          PROP_DEFINE_A(ClusterId, numClusters, 100, --numClusters)
                          PROP_DEFINE_A(float, eps, 0, --eps)
                          PROP_DEFINE_A(float, maxIter, 50, --max_iter)
                  )
                  PROP_DEFINE_A(std::string, outDir, "", --out)
                  PROP_DEFINE_A(uint16_t, numThreads, 4, --numThreads)
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

Result process(std::string const& imageFilename, Weights const& weights, std::string const& spOutPath,
               std::string const& labelOutPath, helper::image::ColorMap const& cmap, ClusterId numClusters, float eps,
               uint32_t maxIter)
{
    std::string filename = boost::filesystem::path(imageFilename).stem().string();
    Result res;
    res.filename = filename;

    // Load image
    FeatureImage features;
    if(!features.read(imageFilename))
    {
        std::cerr << "Unable to read features from \"" << imageFilename << "\"" << std::endl;
        return res;
    }

    // Create energy function
    EnergyFunction energyFun(&weights, numClusters);

    // Do the inference!
    InferenceIterator<EnergyFunction> inference(&energyFun, &features, eps, maxIter);
    auto result = inference.run();

    // Write results to disk
    boost::filesystem::path spPath(spOutPath);
    boost::filesystem::create_directories(spPath);
    boost::filesystem::path labelPath(labelOutPath);
    boost::filesystem::create_directories(labelPath);
    helper::image::writePalettePNG(labelPath.string() + filename + ".png", result.labeling, cmap);
    helper::image::writePalettePNG(spPath.string() + filename + ".png", result.clustering, cmap);
    helper::clustering::write(spPath.string() + filename + ".dat", result.clustering, result.clusters);

    auto errCode = helper::image::writePalettePNG(labelPath.string() + filename + ".png", result.labeling, cmap);
    if(errCode != helper::image::PNGError::Okay)
        return res;
//    if(!helper::image::writeMarginals(labelPath.string() + filename + ".marginals", result.marginals))
//        return res;

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
    properties.read("properties/hseg_infer_batch.info");
    properties.fromCmd(argc, argv);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    Weights weights(properties.dataset.constants.numClasses, properties.dataset.constants.featDim);
    if(!weights.read(properties.param.weights))
    {
        std::cerr << "Couldn't read weights from \"" << properties.param.weights << "\". Using random weights instead." << std::endl;
        weights.randomize();
    }

    helper::image::ColorMap const cmap = helper::image::generateColorMapVOC(256ul);

    // Read in file names to process
    auto filenames = readFileNames(properties.dataset.list);
    if(filenames.empty())
    {
        std::cerr << "No files specified." << std::endl;
        return FILE_LIST_EMPTY;
    }

    boost::filesystem::path spPath(properties.outDir + "/clustering/");
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
    std::deque<std::future<Result>> futures;

    // Iterate all files
    for(auto const& f : filenames)
    {
        std::string const& imageFilename = properties.dataset.path.img + f + properties.dataset.extension.img;
        std::string filename = boost::filesystem::path(imageFilename).stem().string();
        if(boost::filesystem::exists(spPath / (filename + ".dat")) && boost::filesystem::exists(labelPath / (filename + ".png")))
        {
            std::cout << "Skipping " << f << "." << std::endl;
            continue;
        }
        auto&& fut = pool.enqueue(process, imageFilename, weights, spPath.string(), labelPath.string(), cmap, properties.param.numClusters, properties.param.eps, properties.param.maxIter);
        futures.push_back(std::move(fut));

        // Wait for some threads to finish if the queue gets too long
        while(pool.queued() > properties.numThreads * 4)
        {
            Result res = futures.front().get();
            if(!res.okay)
                std::cerr << "Couldn't process image \"" + res.filename + "\"" << std::endl;
            else
                std::cout << "Done with \"" + res.filename + "\"" << std::endl;
            futures.pop_front();
        }
    }

    // Wait for remaining threads to finish
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