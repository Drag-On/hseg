//
// Created by jan on 01.09.16.
//

#include <BaseProperties.h>
#include <Energy/Weights.h>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <helper/image_helper.h>
#include <helper/clustering_helper.h>
#include <Energy/EnergyFunction.h>
#include <Threading/ThreadPool.h>
#include <Energy/LossAugmentedEnergyFunction.h>

PROPERTIES_DEFINE(TrainDistMerge,
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
                  GROUP_DEFINE(train,
                               PROP_DEFINE_A(float, C, 0.1, -C)
                               GROUP_DEFINE(iter,
                                            PROP_DEFINE_A(uint32_t, end, 1000, --end)
                               )
                               GROUP_DEFINE(rate,
                                            PROP_DEFINE_A(float, base, 0.001f, --rate)
                                            PROP_DEFINE_A(float, T, 200, -T)
                               )
                  )
                  GROUP_DEFINE(param,
                               PROP_DEFINE_A(ClusterId, numClusters, 100, --numClusters)
                  )
                  PROP_DEFINE_A(uint32_t, t, 0, -t)
                  PROP_DEFINE_A(std::string, weights, "", --weights)
                  PROP_DEFINE_A(std::string, in, "", --in)
                  PROP_DEFINE_A(std::string, out, "", --out)
                  PROP_DEFINE_A(uint32_t, numThreads, 4, --numThreads)
)

enum ErrorCode
{
    SUCCESS = 0,
    INVALID_FEATURE_WEIGHTS = 1,
    INVALID_FILE_LIST = 2,
    INVALID_SAMPLE = 3,
    CANT_READ_WEIGHTS = 4,
    CANT_WRITE_WEIGHTS = 5,
};

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
    float upperBound = 0;
    Weights gradient{21ul, 512};
    bool valid = false;
};

SampleResult processSample(TrainDistMergeProperties const& properties, std::string filename, Weights const& curWeights)
{
    SampleResult sampleResult;

    std::string const imgFilename = properties.dataset.path.img + filename + properties.dataset.extension.img;
    std::string const gtFilename = properties.dataset.path.gt + filename + properties.dataset.extension.gt;
    std::string const predictionFilename = properties.in + "labeling/" + filename + properties.dataset.extension.gt;
    std::string const clusteringFilename = properties.in + "clustering/" + filename + ".dat";
    std::string const clusteringGtFilename = properties.in + "clustering_gt/" + filename + ".dat";

    // Load images etc...
    FeatureImage features;
    if(!features.read(imgFilename))
    {
        std::cerr << "Unable to read features from \"" << imgFilename << "\"" << std::endl;
        return sampleResult;
    }

    LabelImage gt;
    auto errCode = helper::image::readPalettePNG(gtFilename, gt, nullptr);
    if(errCode != helper::image::PNGError::Okay)
    {
        std::cerr << "Unable to read ground truth from \"" << gtFilename << "\". Error Code: " << (int) errCode << std::endl;
        return sampleResult;
    }

    LabelImage prediction;
    errCode = helper::image::readPalettePNG(predictionFilename, prediction, nullptr);
    if(errCode != helper::image::PNGError::Okay)
    {
        std::cerr << "Unable to read prediction from \"" << predictionFilename << "\". Error Code: " << (int) errCode << std::endl;
        return sampleResult;
    }

    LabelImage clustering, gt_clustering;
    std::vector<Cluster> clusters, gt_clusters;
    if(!helper::clustering::read(clusteringFilename, clustering, clusters))
    {
        std::cerr << "Unable to read clustering from \"" << clusteringFilename << "\"." << std::endl;
        return sampleResult;
    }
    if(!helper::clustering::read(clusteringGtFilename, gt_clustering, gt_clusters))
    {
        std::cerr << "Unable to read ground truth clustering from \"" << clusteringGtFilename << "\"." << std::endl;
        return sampleResult;
    }

    EnergyFunction energy(&curWeights, properties.param.numClusters);

    // Compute energy without weights on the ground truth
    auto gtEnergy = energy.giveEnergyByWeight(features, gt, gt_clustering, gt_clusters);
    // Compute energy without weights on the prediction
    auto predEnergy = energy.giveEnergyByWeight(features, prediction, clustering, clusters);

    // Compute upper bound on this image
    auto gtEnergyCur = curWeights * gtEnergy;
    auto predEnergyCur = curWeights * predEnergy;
    float lossFactor = LossAugmentedEnergyFunction::computeLossFactor(gt, properties.dataset.constants.numClasses);
    float loss = LossAugmentedEnergyFunction::computeLoss(prediction, gt, lossFactor, properties.dataset.constants.numClasses);
    sampleResult.upperBound = (loss - predEnergyCur) + gtEnergyCur;

    // Compute gradient for this sample
    gtEnergy -= predEnergy;
    sampleResult.gradient = gtEnergy;

    sampleResult.valid = true;
    return sampleResult;
}

int main(int argc, char* argv[])
{
    // Read properties
    TrainDistMergeProperties properties;
    properties.read("properties/hseg_train_dist_merge.info");
    properties.fromCmd(argc, argv);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    Label const numClasses = properties.dataset.constants.numClasses;
    uint32_t featDim = properties.dataset.constants.featDim;

    Weights curWeights(numClasses, featDim);
    if(!curWeights.read(properties.weights) && properties.t != 0)
    {
        std::cerr << "Couldn't read current weights from \"" << properties.weights << "\"" << std::endl;
        return CANT_READ_WEIGHTS;
    }

    // Read in list of files
    std::vector<std::string> list = readFileNames(properties.dataset.list);
    if(list.empty())
    {
        std::cerr << "File list \"" << properties.dataset.list << "\" is empty." << std::endl;
        return INVALID_FILE_LIST;
    }

    // Iterate over all predictions
    size_t N = list.size();
    size_t t = properties.t;
    Weights sum(numClasses, featDim);
    ThreadPool pool(properties.numThreads);
    std::vector<std::future<SampleResult>> futures;
    // Iterate over all images
    for (size_t n = 0; n < N; ++n)
    {
        auto&& fut = pool.enqueue(processSample, properties, list[n],curWeights);
        futures.push_back(std::move(fut));
    }

    Cost trainingEnergy = 0;
    for(size_t n = 0; n < futures.size(); ++n)
    {
        auto sampleResult = futures[n].get();
        if(!sampleResult.valid)
        {
            std::cerr << "Sample result was invalid. Cannot continue." << std::endl;
            return INVALID_SAMPLE;
        }

        sum += sampleResult.gradient;
        trainingEnergy += sampleResult.upperBound;

        std::cout << "<<< " << std::setw(4) << t << "/" << std::setw(4) << n << " >>>\t" << sampleResult.upperBound << std::endl;
    }

    // Compute step size
    float stepSize = properties.train.rate.base / (1 + t / properties.train.rate.T);

    // Show current training energy
    trainingEnergy *= properties.train.C / N;
    trainingEnergy += curWeights.sqNorm() / 2.f;
    std::cout << "Current upper bound: " << trainingEnergy << std::endl;
    boost::filesystem::path energyFilePath(properties.out);
    energyFilePath.remove_filename();
    std::ofstream out(energyFilePath.string() + "/training_energy.txt", std::ios::out | std::ios::app);
    if(out.is_open())
    {
        out.precision(std::numeric_limits<float>::max_digits10);
        out << std::setw(4) << properties.t << "\t";
        out << std::setw(12) << trainingEnergy << "\t";
        out << std::setw(12) << stepSize << std::endl;
        out.close();
    }
    else
        std::cerr << "Couldn't write current training energy to file " << energyFilePath << std::endl;

    // Update step
    sum *= properties.train.C / N;
    sum += curWeights;
    sum *= stepSize;
    curWeights -= sum;

    // Project onto the feasible set
    curWeights.clampToFeasible();

    if(!curWeights.write(properties.out))
    {
        std::cerr << "Couldn't write weights to file " << properties.out << std::endl;
        return CANT_WRITE_WEIGHTS;
    }
    if(!curWeights.write(energyFilePath.string() + "/weights/" + std::to_string(properties.t + 1) + ".dat"))
    {
        std::cerr << "Couldn't write weights to file " << properties.out << std::endl;
        return CANT_WRITE_WEIGHTS;
    }

    return SUCCESS;
}