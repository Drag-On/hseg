//
// Created by jan on 29.08.16.
//

#include <BaseProperties.h>
#include <Energy/Weights.h>
#include <Energy/LossAugmentedEnergyFunction.h>
#include <helper/image_helper.h>
#include <Inference/InferenceIterator.h>
#include <Threading/ThreadPool.h>
#include <Energy/IStepSizeRule.h>
#include <Energy/AdamStepSizeRule.h>


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
                               PROP_DEFINE_A(bool, useClusterLoss, true, --useClusterLoss)
                               GROUP_DEFINE(iter,
                                            PROP_DEFINE_A(uint32_t, start, 0, --start)
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
                               PROP_DEFINE_A(bool, usePairwise, false, --usePairwise)
                               PROP_DEFINE_A(float, eps, 0, --eps)
                               PROP_DEFINE_A(float, maxIter, 50, --max_iter)
                  )
                  PROP_DEFINE_A(std::string, in, "", -i)
                  PROP_DEFINE_A(std::string, out, "", -o)
                  PROP_DEFINE_A(std::string, outDir, "", --outDir)
                  PROP_DEFINE_A(std::string, log, "train.log", --log)
                  PROP_DEFINE_A(uint32_t, numThreads, 4, --numThreads)
                  PROP_DEFINE_A(std::string, propertiesFile, "properties/hseg_train.info", -p)
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
    uint32_t numIter = 0;
    uint32_t numIterGt = 0;
    std::string filename;
};

SampleResult processSample(std::string const& filename, Weights const& curWeights, TrainProperties const& properties)
{
    SampleResult sampleResult;
    sampleResult.filename = filename;

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
    gt.rescale(features.width(), features.height(), false);

    // Crop to valid region
    cv::Rect bb = helper::image::computeValidBox(gt, properties.dataset.constants.numClasses);
    FeatureImage features_cropped(bb.width, bb.height, features.dim());
    LabelImage gt_cropped(bb.width, bb.height);
    for(Coord x = bb.x; x < bb.width; ++x)
    {
        for (Coord y = bb.y; y < bb.height; ++y)
        {
            gt_cropped.at(x - bb.x, y - bb.y) = gt.at(x, y);
            features_cropped.at(x - bb.x, y - bb.y) = features.at(x, y);
        }
    }

    gt = gt_cropped;
    features = features_cropped;

    if(gt.height() == 0 || gt.width() == 0 || gt.height() != features.height() || gt.width() != features.width())
    {
        std::cerr << "Invalid ground truth or features. Dimensions: (" << gt.width() << "x" << gt.height() << ") vs. ("
                  << features.width() << "x" << features.height() << ")." << std::endl;
        return sampleResult;
    }


    // Find latent variables that best explain the ground truth
    EnergyFunction energy(&curWeights, properties.param.numClusters, properties.param.usePairwise);
    InferenceIterator<EnergyFunction> gtInference(&energy, &features, properties.param.eps, properties.param.maxIter);
    InferenceResult gtResult = gtInference.runOnGroundTruth(gt);
    sampleResult.numIterGt = gtResult.numIter;

    // Predict with loss-augmented energy
    LossAugmentedEnergyFunction lossEnergy(&curWeights, &gt, properties.param.numClusters, properties.param.usePairwise, properties.train.useClusterLoss);
    InferenceIterator<LossAugmentedEnergyFunction> inference(&lossEnergy, &features, properties.param.eps, properties.param.maxIter);
    InferenceResult result = inference.run();
    sampleResult.numIter = result.numIter;

    // Compute energy without weights on the ground truth
    auto gtEnergy = energy.giveEnergyByWeight(features, gt, gtResult.clustering, gtResult.clusters);
    // Compute energy without weights on the prediction
    auto predEnergy = energy.giveEnergyByWeight(features, result.labeling, result.clustering, result.clusters);

    //std::cout << gtEnergy.sum() << ", " << predEnergy.sum() << ", " << curWeights.sum() << std::endl;

    // Compute upper bound on this image
    auto gtEnergyCur = curWeights * gtEnergy;
    auto predEnergyCur = curWeights * predEnergy;
    float lossFactor = LossAugmentedEnergyFunction::computeLossFactor(gt, properties.dataset.constants.numClasses);
    float loss = LossAugmentedEnergyFunction::computeLoss(result.labeling, result.clustering, gt, result.clusters,
                                                          lossFactor, properties.dataset.constants.numClasses, properties.train.useClusterLoss);
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
    NO_VALID_SAMPLES,
};

int main(int argc, char** argv)
{
    // Read properties
    TrainProperties properties;
    properties.fromCmd(argc, argv);
    properties.read(properties.propertiesFile);
    properties.fromCmd(argc, argv); // This is so the property file location can be read via command line. However,
                                    // the command line arguments should overwrite anything written in the file,
                                    // therefore read it in again.
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    Weights curWeights(properties.dataset.constants.numClasses, properties.dataset.constants.featDim);
    if(!curWeights.read(properties.in))
        std::cout << "Couldn't read in initial weights from \"" << properties.in << "\". Using zero." << std::endl;

    std::string weightCopyFilename = properties.outDir + std::to_string(properties.train.iter.start) + ".dat";
    if(!curWeights.write(weightCopyFilename))
    {
        std::cerr << "Couldn't write initial weights to file \"" << weightCopyFilename << "\"" << std::endl;
        return CANT_WRITE_RESULT_BACKUP;
    }

    // Load filenames of all images
    std::vector<std::string> filenames = readFileNames(properties.dataset.list);
    if (filenames.empty())
    {
        std::cerr << "File list is empty!" << std::endl;
        return FILE_LIST_EMPTY;
    }
    uint32_t T = properties.train.iter.end - properties.train.iter.start;

    ThreadPool pool(properties.numThreads);
    std::deque<std::future<SampleResult>> futures;

    // Initialize step size rule
    auto pStepSizeRule = std::make_unique<AdamStepSizeRule>(properties.train.rate.alpha,
                                                            properties.train.rate.beta1,
                                                            properties.train.rate.beta2,
                                                            properties.train.rate.eps,
                                                            properties.dataset.constants.numClasses,
                                                            properties.dataset.constants.featDim,
                                                            properties.train.iter.start);
    if(!pStepSizeRule->read(properties.outDir, properties.train.iter.start))
        std::cout << "Couldn't read in initial step size meta data from \"" << properties.outDir << "\". Using default." << std::endl;

    auto mode = std::ios::out;
    if(properties.train.iter.start == 0)
        mode |= std::ios::trunc;
    else
        mode |= std::ios::app;
    std::ofstream log(properties.log, mode);
    if(!log.is_open())
    {
        std::cerr << "Cannot write to log file \"" << properties.log << "\"" << std::endl;
        return CANT_WRITE_LOG;
    }

    // Print header to log file
    if (properties.train.iter.start == 0)
    {
        log << std::setw(4) << "iter" << "\t;"
            << std::setw(12) << "objective" << "\t;"
            << std::setw(12) << "regularizer" << "\t;"
            << std::setw(12) << "upper bound" << "\t;"
            << std::setw(12) << "unary mean" << "\t;"
            << std::setw(12) << "unary stdev" << "\t;"
            << std::setw(12) << "unary max" << "\t;"
            << std::setw(12) << "unary min" << "\t;"
            << std::setw(12) << "unary mag" << "\t;"
            << std::setw(12) << "pair mean" << "\t;"
            << std::setw(12) << "pair stdev" << "\t;"
            << std::setw(12) << "pair max" << "\t;"
            << std::setw(12) << "pair min" << "\t;"
            << std::setw(12) << "pair mag" << "\t;"
            << std::setw(12) << "label mean" << "\t;"
            << std::setw(12) << "label stdev" << "\t;"
            << std::setw(12) << "label max" << "\t;"
            << std::setw(12) << "label min" << "\t;"
            << std::setw(12) << "label mag" << "\t;"
            << std::setw(12) << "feat mean" << "\t;"
            << std::setw(12) << "feat stdev" << "\t;"
            << std::setw(12) << "feat max" << "\t;"
            << std::setw(12) << "feat min" << "\t;"
            << std::setw(12) << "feat mag" << "\t;"
            << std::setw(12) << "total mean" << "\t;"
            << std::setw(12) << "total stdev" << "\t;"
            << std::setw(12) << "total max" << "\t;"
            << std::setw(12) << "total min" << "\t;"
            << std::setw(12) << "total mag" << "\t;";
    }

    // Iterate T times
    for(uint32_t t = properties.train.iter.start; t < T; ++t)
    {
        uint32_t N = 0;
        Weights sum(properties.dataset.constants.numClasses, properties.dataset.constants.featDim); // All zeros
        Cost iterationEnergy = 0;
        futures.clear();

        // Iterate over all images
        for (std::string const filename : filenames)
        {
            auto&& fut = pool.enqueue(processSample, filename, curWeights, properties);
            futures.push_back(std::move(fut));

            // Wait for some threads to finish if the queue gets too long
            while(pool.queued() > properties.numThreads)
            {
                auto sampleResult = futures.front().get();
                if(!sampleResult.valid)
                {
                    std::cerr << "Sample result was invalid. Cannot continue." << std::endl;
                    return INFERRED_INVALID;
                }

                // Filter out bad results
                if (sampleResult.upperBound >= 0)
                {
                    sum += sampleResult.gradient;
                    iterationEnergy += sampleResult.upperBound;
                    N++;
                }

                std::cout << "> " << std::setw(4) << t << ": " << std::setw(30) << sampleResult.filename << "\t"
                          << std::setw(12) << sampleResult.upperBound << "\t"
                          << std::setw(2) << sampleResult.numIter << "\t"
                          << std::setw(2) << sampleResult.numIterGt << std::endl;
                futures.pop_front();
            }
        }

        // Wait for remaining threads to finish
        for(size_t n = 0; n < futures.size(); ++n)
        {
            auto sampleResult = futures[n].get();
            if(!sampleResult.valid)
            {
                std::cerr << "Sample result \"" << sampleResult.filename << "\" was invalid. Cannot continue." << std::endl;
                return INFERRED_INVALID;
            }

            std::cout << "> " << std::setw(4) << t << ": " << std::setw(30) << sampleResult.filename << "\t"
                      << std::setw(12) << sampleResult.upperBound << "\t"
                      << std::setw(2) << sampleResult.numIter << "\t"
                      << std::setw(2) << sampleResult.numIterGt << std::endl;

            // Filter out bad results
            if (sampleResult.upperBound >= 0)
            {
                sum += sampleResult.gradient;
                iterationEnergy += sampleResult.upperBound;
                N++;
            }
        }

        if(N <= 0)
        {
            std::cerr << "There were no valid samples. Terminating..." << std::endl;
            return NO_VALID_SAMPLES;
        }

        // Show current training energy
        iterationEnergy *= properties.train.C / N;
        auto upperBoundCost = iterationEnergy;
        auto regularizerCost = curWeights.regularized().sqNorm() / 2.f;
        iterationEnergy += regularizerCost;
        std::cout << "Current training energy: " << regularizerCost << " + " << upperBoundCost << " = " << iterationEnergy << std::endl;

        // Log results of last iteration
        auto stats = curWeights.computeStats();
        log << std::setw(4) << t << "\t;"
            << std::setw(12) << iterationEnergy << "\t;"
            << std::setw(12) << regularizerCost << "\t;"
            << std::setw(12) << upperBoundCost << "\t;"
            << std::setw(12) << stats.unary.mean << "\t;"
            << std::setw(12) << stats.unary.stdev << "\t;"
            << std::setw(12) << stats.unary.max << "\t;"
            << std::setw(12) << stats.unary.min << "\t;"
            << std::setw(12) << stats.unary.mag << "\t;"
            << std::setw(12) << stats.pairwise.mean << "\t;"
            << std::setw(12) << stats.pairwise.stdev << "\t;"
            << std::setw(12) << stats.pairwise.max << "\t;"
            << std::setw(12) << stats.pairwise.min << "\t;"
            << std::setw(12) << stats.pairwise.mag << "\t;"
            << std::setw(12) << stats.label.mean << "\t;"
            << std::setw(12) << stats.label.stdev << "\t;"
            << std::setw(12) << stats.label.max << "\t;"
            << std::setw(12) << stats.label.min << "\t;"
            << std::setw(12) << stats.label.mag << "\t;"
            << std::setw(12) << stats.feature.mean << "\t;"
            << std::setw(12) << stats.feature.stdev << "\t;"
            << std::setw(12) << stats.feature.max << "\t;"
            << std::setw(12) << stats.feature.min << "\t;"
            << std::setw(12) << stats.feature.mag << "\t;"
            << std::setw(12) << stats.total.mean << "\t;"
            << std::setw(12) << stats.total.stdev << "\t;"
            << std::setw(12) << stats.total.max << "\t;"
            << std::setw(12) << stats.total.min << "\t;"
            << std::setw(12) << stats.total.mag << "\t;";

        // Compute gradient
        sum *= properties.train.C / N;
        sum += curWeights.regularized();
        // ... and update
        pStepSizeRule->update(curWeights, sum);

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
        pStepSizeRule->write(properties.outDir);
    }

    log.close();

    return SUCCESS;
}