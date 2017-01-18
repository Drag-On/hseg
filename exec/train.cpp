//
// Created by jan on 29.08.16.
//

#include <BaseProperties.h>
#include <Energy/Weights.h>
#include <Energy/LossAugmentedEnergyFunction.h>
#include <helper/image_helper.h>
#include <Inference/InferenceIterator.h>
#include <Threading/ThreadPool.h>


PROPERTIES_DEFINE(Train,
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
                                            PROP_DEFINE_A(uint32_t, start, 0, --start)
                                            PROP_DEFINE_A(uint32_t, end, 1000, --end)
                               )
                               GROUP_DEFINE(rate,
                                            PROP_DEFINE_A(float, base, 0.001f, --rate)
                                            PROP_DEFINE_A(float, T, 200, -T)
                               )
                  )
                  GROUP_DEFINE(param,
                               PROP_DEFINE_A(ClusterId, numClusters, 100, --numClusters)
                               PROP_DEFINE_A(float, eps, 0, --eps)
                  )
                  PROP_DEFINE_A(std::string, in, "", -i)
                  PROP_DEFINE_A(std::string, out, "", -o)
                  PROP_DEFINE_A(std::string, outDir, "", --outDir)
                  PROP_DEFINE_A(std::string, log, "train.log", --log)
                  PROP_DEFINE_A(uint32_t, numThreads, 4, --numThreads)
)

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
    Weights gradient{21ul, 512};
    Cost upperBound = 0;
    bool valid = false;
};

SampleResult processSample(std::string const& filename, Weights const& curWeights, TrainProperties const& properties)
{
    SampleResult sampleResult;

    // Load images etc...
    std::string imgFilename = properties.dataset.path.img + filename + properties.dataset.extension.img;
    std::string gtFilename = properties.dataset.path.gt + filename + properties.dataset.extension.gt;

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

    if(features.height() != gt.height() || features.width() != gt.width())
    {
        std::cerr << "Dimensions of \"" << imgFilename << "\" and \"" << gtFilename << "\" don't match." << std::endl;
        return sampleResult;
    }

    // Find latent variables that best explain the ground truth
    EnergyFunction energy(&curWeights, properties.param.numClusters);
    InferenceIterator<EnergyFunction> gtInference(&energy, &features, properties.param.eps);
    InferenceResult gtResult = gtInference.runOnGroundTruth(gt);

    // Predict with loss-augmented energy
    LossAugmentedEnergyFunction lossEnergy(&curWeights, &gt, properties.param.numClusters);
    InferenceIterator<LossAugmentedEnergyFunction> inference(&lossEnergy, &features, properties.param.eps);
    InferenceResult result = inference.run();

    // Compute energy without weights on the ground truth
    auto gtEnergy = energy.giveEnergyByWeight(features, gt, gtResult.clustering, gtResult.clusters);
    // Compute energy without weights on the prediction
    auto predEnergy = energy.giveEnergyByWeight(features, result.labeling, result.clustering, result.clusters);

    //std::cout << gtEnergy.sum() << ", " << predEnergy.sum() << ", " << curWeights.sum() << std::endl;

    // Compute upper bound on this image
    auto gtEnergyCur = curWeights * gtEnergy;
    auto predEnergyCur = curWeights * predEnergy;
    float lossFactor = LossAugmentedEnergyFunction::computeLossFactor(gt, properties.dataset.constants.numClasses);
    float loss = LossAugmentedEnergyFunction::computeLoss(result.labeling, gt, lossFactor, properties.dataset.constants.numClasses);
    sampleResult.upperBound = (loss - predEnergyCur) + gtEnergyCur;

    //std::cout << "Upper bound: (" << loss << " - " << predEnergyCur << ") + " << gtEnergyCur << " = " << loss - predEnergyCur << " + " << gtEnergyCur << " = " << sampleResult.upperBound << std::endl;

    // Compute gradient for this sample
    gtEnergy -= predEnergy;
    sampleResult.gradient = gtEnergy;

    sampleResult.valid = true;
    return sampleResult;
}

enum ERROR_CODE
{
    SUCCESS=0,
    FILE_LIST_EMPTY,
    CANT_WRITE_LOG,
    INFERRED_INVALID,
    CANT_WRITE_RESULT,
    CANT_WRITE_RESULT_BACKUP,
};

int main(int argc, char** argv)
{
    // Read properties
    TrainProperties properties;
    properties.read("properties/hseg_train.info");
    properties.fromCmd(argc, argv);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

//    helper::image::ColorMap const cmap = helper::image::generateColorMapVOC(256ul);

    Weights curWeights(properties.dataset.constants.numClasses, properties.dataset.constants.featDim);
    if(!curWeights.read(properties.in))
        std::cout << "Couldn't read in initial weights from \"" << properties.in << "\". Using zero." << std::endl;

    std::string weightCopyFilename = properties.outDir + std::to_string(properties.train.iter.start) + ".dat";
    if(!curWeights.write(weightCopyFilename))
    {
        std::cerr << "Couldn't write initial weights to file \"" << weightCopyFilename << "\"" << std::endl;
        return CANT_WRITE_RESULT_BACKUP;
    }

    std::random_device rd;
    std::default_random_engine random(rd());

    // Load filenames of all images
    std::vector<std::string> filenames = readFileNames(properties.dataset.list);
    if (filenames.empty())
    {
        std::cerr << "File list is empty!" << std::endl;
        return FILE_LIST_EMPTY;
    }
    uint32_t T = properties.train.iter.end - properties.train.iter.start;
    uint32_t N = filenames.size();


    ThreadPool pool(properties.numThreads);
    std::vector<std::future<SampleResult>> futures;

//    Cost lastTrainingEnergy = std::numeric_limits<Cost>::max();
    Cost learningRate = properties.train.rate.base;
    Weights lastWeights = curWeights;
    Weights lastGradient = curWeights;

    std::ofstream log(properties.log);
    if(!log.is_open())
    {
        std::cerr << "Cannot write to log file \"" << properties.log << "\"" << std::endl;
        return CANT_WRITE_LOG;
    }

    // Iterate T times
    for(uint32_t t = properties.train.iter.start; t < T; ++t)
    {
        Weights sum(properties.dataset.constants.numClasses, properties.dataset.constants.featDim); // All zeros
        Cost iterationEnergy = 0;
        futures.clear();

        // Iterate over all images
        for (size_t n = 0; n < N; ++n)
        {
            std::string const filename = filenames[n];

            auto&& fut = pool.enqueue(processSample, filename, curWeights, properties);
            futures.push_back(std::move(fut));
        }

        for(size_t n = 0; n < futures.size(); ++n)
        {
            auto sampleResult = futures[n].get();
            if(!sampleResult.valid)
            {
                std::cerr << "Sample result was invalid. Cannot continue." << std::endl;
                return INFERRED_INVALID;
            }

            std::cout << "<<< " << std::setw(4) << t << "/" << std::setw(4) << n << " >>>\t" << sampleResult.upperBound << std::endl;

            sum += sampleResult.gradient;
            iterationEnergy += sampleResult.upperBound;
        }

        // Show current training energy
        iterationEnergy *= properties.train.C / N;
        auto upperBoundCost = iterationEnergy;
        auto regularizerCost = curWeights.sqNorm() / 2.f;
        iterationEnergy += regularizerCost;
        std::cout << "Current training energy: " << regularizerCost << " + " << upperBoundCost << " = " << iterationEnergy << std::endl;

        // Compute learning rate
        learningRate = properties.train.rate.base / (1 + t / properties.train.rate.T);

        lastWeights = curWeights;
//        lastTrainingEnergy = iterationEnergy;

        log << std::setw(4) << t << "\t" << std::setw(12) << iterationEnergy << "\t" << std::setw(12) << learningRate << std::endl;

        // Update step
        sum *= properties.train.C / N;
        sum += curWeights;
        lastGradient = sum;
        sum *= learningRate;
        curWeights -= sum;

        // Project onto the feasible set
        curWeights.clampToFeasible();

        if (!curWeights.write(properties.out))
        {
            std::cerr << "Couldn't write weights to file \"" << properties.out << "\"" << std::endl;
            log.close();
            return CANT_WRITE_RESULT;
        }
        std::string weightCopyFilename = properties.outDir + std::to_string(t + 1) + ".dat";
        if(!curWeights.write(weightCopyFilename))
        {
            std::cerr << "Couldn't write weights to file \"" << weightCopyFilename << "\"" << std::endl;
            log.close();
            return CANT_WRITE_RESULT_BACKUP;
        }
    }

    log.close();

    return SUCCESS;
}