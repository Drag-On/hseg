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
                                            PROP_DEFINE_A(float, alpha, 0.001f, --alpha)
                                            PROP_DEFINE_A(float, beta1, 0.9f, --beta1)
                                            PROP_DEFINE_A(float, beta2, 0.999f, --beta2)
                                            PROP_DEFINE_A(float, eps, 10e-8f, --rate_eps)
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
    std::string filename;
};

SampleResult processSample(TrainDistMergeProperties const& properties, std::string filename, Weights const& curWeights)
{
    SampleResult sampleResult;
    sampleResult.filename = filename;

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
    float loss = LossAugmentedEnergyFunction::computeLoss(prediction, clustering, gt, clusters, lossFactor,
                                                          properties.dataset.constants.numClasses);
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

    boost::filesystem::path energyFilePath(properties.out);
    energyFilePath.remove_filename();

    // Write initial weights to file
    if(properties.t == 0)
    {
        std::string backupWeightsFile = energyFilePath.string() + "/weights/0.dat";
        if(!curWeights.write(backupWeightsFile))
        {
            std::cerr << "Couldn't write weights to file " << backupWeightsFile << std::endl;
            return CANT_WRITE_WEIGHTS;
        }
    }

    // Read in list of files
    std::vector<std::string> list = readFileNames(properties.dataset.list);
    if(list.empty())
    {
        std::cerr << "File list \"" << properties.dataset.list << "\" is empty." << std::endl;
        return INVALID_FILE_LIST;
    }

    // Initialize moment vectors (adam step size rule)
    Weights curFirstMomentVector(properties.dataset.constants.numClasses, properties.dataset.constants.featDim);
    Weights curSecondMomentVector(properties.dataset.constants.numClasses, properties.dataset.constants.featDim);
    float const adam_alpha = properties.train.rate.alpha;
    float const adam_beta1 = properties.train.rate.beta1;
    float const adam_beta2 = properties.train.rate.beta2;
    float const adam_eps = properties.train.rate.eps;

    // Iterate over all predictions
    size_t N = list.size();
    size_t t = properties.t;
    Weights sum(numClasses, featDim);
    Cost trainingEnergy = 0;
    ThreadPool pool(properties.numThreads);
    std::deque<std::future<SampleResult>> futures;
    // Iterate over all images
    for (size_t n = 0; n < N; ++n)
    {
        auto&& fut = pool.enqueue(processSample, properties, list[n],curWeights);
        futures.push_back(std::move(fut));

        // Wait for some threads to finish if the queue gets too long
        while(pool.queued() > properties.numThreads * 4)
        {
            auto sampleResult = futures.front().get();
            if(!sampleResult.valid)
            {
                std::cerr << "Sample result was invalid. Cannot continue." << std::endl;
                return INVALID_SAMPLE;
            }

            sum += sampleResult.gradient;
            trainingEnergy += sampleResult.upperBound;

            std::cout << "<<< " << std::setw(4) << t << " / " << sampleResult.filename << " >>>\t" << sampleResult.upperBound << std::endl;
            futures.pop_front();
        }
    }

    // Wait for remaining threads to finish
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

        std::cout << "<<< " << std::setw(4) << t << " / " << sampleResult.filename << " >>>\t" << sampleResult.upperBound << std::endl;
    }

    // Show current training energy
    trainingEnergy *= properties.train.C / N;
    trainingEnergy += curWeights.sqNorm() / 2.f;
    std::cout << "Current upper bound: " << trainingEnergy << std::endl;
    std::ofstream out(energyFilePath.string() + "/training_energy.txt", std::ios::out | std::ios::app);
    if(out.is_open())
    {
        out.precision(std::numeric_limits<float>::max_digits10);
        out << std::setw(4) << properties.t << "\t";
        out << std::setw(12) << trainingEnergy << std::endl;
        out.close();
    }
    else
        std::cerr << "Couldn't write current training energy to file " << energyFilePath << std::endl;

    // Compute gradient
    sum *= properties.train.C / N;
    sum += curWeights;

    // Update biased 1st and 2nd moment estimates
    curFirstMomentVector = curFirstMomentVector * adam_beta1 + sum * (1 - adam_beta2);
    sum.squareElements();
    curSecondMomentVector = curSecondMomentVector * adam_beta2 + sum * (1 - adam_beta2);

    // Update weights
    float const curAlpha =
            adam_alpha * std::sqrt(1 - std::pow(adam_beta2, t + 1)) / (1 - std::pow(adam_beta1, t + 1));
    auto sqrtSecondMomentVector = curSecondMomentVector;
    sqrtSecondMomentVector.sqrt();
    curWeights -= (curFirstMomentVector * curAlpha) / (sqrtSecondMomentVector + adam_eps);

    // Project onto the feasible set
    curWeights.clampToFeasible();

    if(!curWeights.write(properties.out))
    {
        std::cerr << "Couldn't write weights to file " << properties.out << std::endl;
        return CANT_WRITE_WEIGHTS;
    }
    std::string backupWeightsFile = energyFilePath.string() + "/weights/" + std::to_string(properties.t + 1) + ".dat";
    if(!curWeights.write(backupWeightsFile))
    {
        std::cerr << "Couldn't write weights to file " << backupWeightsFile << std::endl;
        return CANT_WRITE_WEIGHTS;
    }

    return SUCCESS;
}