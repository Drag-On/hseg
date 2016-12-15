//
// Created by jan on 29.08.16.
//

#include <BaseProperties.h>
#include <Energy/WeightsVec.h>
#include <Energy/UnaryFile.h>
#include <Energy/LossAugmentedEnergyFunction.h>
#include <helper/image_helper.h>
#include <Inference/InferenceIterator.h>
#include <Threading/ThreadPool.h>
#include <Energy/feature_weights.h>


PROPERTIES_DEFINE(Train,
                  PROP_DEFINE(size_t, numClusters, 300)
                  PROP_DEFINE(size_t, numIter, 100)
                  PROP_DEFINE_A(size_t, startIter, 0, -t)
                  PROP_DEFINE(std::string, learningRate, "Fixed")
                  GROUP_DEFINE(FixedLearningRate,
                      PROP_DEFINE(float, rate, 0.0001f)
                  )
                  GROUP_DEFINE(DiminishingLearningRate,
                      PROP_DEFINE(float, base, 0.0001f)
                      PROP_DEFINE(float, T, 1)
                  )
                  GROUP_DEFINE(BoldDriverLearningRate,
                      PROP_DEFINE(float, base, 0.0001f)
                      PROP_DEFINE(float, increase, 1.05f)
                      PROP_DEFINE(float, decrease, 0.5f)
                      PROP_DEFINE(float, margin, 1e-10f)
                  )
                  PROP_DEFINE(float, C, 1.f)
                  PROP_DEFINE(float, pairwiseSigmaSq, 0.05f)
                  PROP_DEFINE(std::string, featureWeightFile, "")
                  PROP_DEFINE(size_t, imagesPerIter, 0)
                  PROP_DEFINE(std::string, imageListFile, "")
                  PROP_DEFINE(std::string, imageBasePath, "")
                  PROP_DEFINE(std::string, groundTruthBasePath, "")
                  PROP_DEFINE(std::string, unaryBasePath, "")
                  PROP_DEFINE(std::string, imageExtension, ".jpg")
                  PROP_DEFINE(std::string, gtExtension, ".png")
                  PROP_DEFINE(std::string, in, "")
                  PROP_DEFINE(std::string, outDir, "out/")
                  PROP_DEFINE(std::string, outFile, "weights.dat")
                  PROP_DEFINE(std::string, log, "train.log")
                  PROP_DEFINE(size_t, numThreads, 4)
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
    WeightsVec energyDiff{21ul, false};
    float trainingEnergy = 0;
    bool valid = false;
};

SampleResult processSample(std::string const& colorImgFilename, std::string const& gtImageFilename,
                           std::string const& unaryFilename, TrainProperties const& properties,
                           helper::image::ColorMap const& cmap, size_t numClasses, size_t numClusters,
                           WeightsVec const& curWeights, Matrix5f const& featureWeights)
{
    SampleResult sampleResult;

    // Load images etc...
    RGBImage rgbImage, groundTruthRGB, groundTruthSpRGB;
    rgbImage.read(properties.imageBasePath + colorImgFilename + properties.imageExtension);
    groundTruthRGB.read(properties.groundTruthBasePath + gtImageFilename + properties.gtExtension);
    if (rgbImage.width() != groundTruthRGB.width() || rgbImage.height() != groundTruthRGB.height())
    {
        std::cerr << "Image " << colorImgFilename << " and its ground truth don't match." << std::endl;
        return sampleResult;
    }
    CieLabImage cieLabImage = rgbImage.getCieLabImg();
    LabelImage groundTruth = helper::image::decolorize(groundTruthRGB, cmap);

    UnaryFile unary(properties.unaryBasePath + unaryFilename + "_prob.dat");
    if(unary.width() != rgbImage.width() || unary.height() != rgbImage.height() || unary.classes() != numClasses)
    {
        std::cerr << "Invalid unary scores " << unaryFilename << std::endl;
        return sampleResult;
    }

    // Find superpixels that best explain the ground truth
    EnergyFunction trainingEnergy(unary, curWeights, properties.pairwiseSigmaSq, featureWeights);
    Clusterer<EnergyFunction> clusterer(trainingEnergy);
    clusterer.run(numClusters, numClasses, cieLabImage, groundTruth);
    LabelImage const& bestSp = clusterer.clustership();

    // Predict with loss-augmented energy
    LossAugmentedEnergyFunction energy(unary, curWeights, properties.pairwiseSigmaSq, featureWeights, groundTruth);
    InferenceIterator<LossAugmentedEnergyFunction> inference(energy, properties.numClusters, numClasses, cieLabImage);
    InferenceResult result = inference.run();

    /*cv::imshow("sp", static_cast<cv::Mat>(helper::image::colorize(result.superpixels, cmap2)));
    cv::imshow("prediction", static_cast<cv::Mat>(helper::image::colorize(result.labeling, cmap)));
    cv::imshow("unary", static_cast<cv::Mat>(helper::image::colorize(unary.maxLabeling(), cmap)));
    cv::imshow("gt", static_cast<cv::Mat>(groundTruthRGB));
    cv::imshow("gt sp", static_cast<cv::Mat>(groundTruthSpRGB));
    cv::waitKey();*/

    auto clusters = Clusterer<EnergyFunction>::computeClusters(result.superpixels, cieLabImage, result.labeling,
                                                               numClusters, numClasses, trainingEnergy);
    auto predEnergyCur = trainingEnergy.giveEnergy(result.labeling, cieLabImage, result.superpixels, clusters);
    sampleResult.trainingEnergy -= predEnergyCur;
    auto gtClusters = Clusterer<EnergyFunction>::computeClusters(bestSp, cieLabImage, groundTruth, numClusters,
                                                                 numClasses, trainingEnergy);
    auto gtEnergyCur = trainingEnergy.giveEnergy(groundTruth, cieLabImage, bestSp, gtClusters);
    sampleResult.trainingEnergy += gtEnergyCur;

    // Compute loss
    float lossFactor = LossAugmentedEnergyFunction::computeLossFactor(groundTruth, numClasses);
    float loss = LossAugmentedEnergyFunction::computeLoss(result.labeling, groundTruth, lossFactor, numClasses);
    sampleResult.trainingEnergy += loss;

    if(sampleResult.trainingEnergy <= 0)
    {
        std::cerr << "Training energy was negative: " << sampleResult.trainingEnergy << " (Loss: " << loss
                  << ", prediction: " << predEnergyCur << ", ground truth: " << gtEnergyCur << ")" << std::endl;
    }

    // Compute energy without weights on the ground truth
    auto gtEnergy = trainingEnergy.giveEnergyByWeight(groundTruth, cieLabImage, bestSp, gtClusters);
    // Compute energy without weights on the prediction
    auto predEnergy = trainingEnergy.giveEnergyByWeight(result.labeling, cieLabImage, result.superpixels, clusters);
    // Compute energy difference
    gtEnergy -= predEnergy;

    sampleResult.energyDiff = gtEnergy;
    sampleResult.valid = true;
    return sampleResult;
}

int main(int argc, char** argv)
{
    // Read properties
    TrainProperties properties;
    properties.read("properties/training.info");
    properties.fromCmd(argc, argv);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    size_t const numClasses = 21;
    size_t const numClusters = properties.numClusters;
    helper::image::ColorMap const cmap = helper::image::generateColorMapVOC(std::max(256ul, numClasses));
    WeightsVec curWeights(numClasses, 100, 100, 100, 100);
    if(!curWeights.read(properties.in))
        std::cout << "Couldn't read in weights to start from. Using default weights." << std::endl;

    std::cout << "====================" << std::endl;
    std::cout << "Initial weights:" << std::endl;
    std::cout << curWeights << std::endl;

    std::random_device rd;
    std::default_random_engine random(rd());

    // Load filenames of all images
    std::vector<std::string> filenames = readFileNames(properties.imageListFile);
    if (filenames.empty())
    {
        std::cerr << "File lists is empty!" << std::endl;
        return 1;
    }
    if (properties.imagesPerIter == 0)
        properties.imagesPerIter = filenames.size();
    size_t T = properties.numIter;
    size_t N = std::min(filenames.size(), properties.imagesPerIter);

    Matrix5f featureWeights = readFeatureWeights(properties.featureWeightFile);
    featureWeights = featureWeights.inverse();
    std::cout << "Used feature weights: " << std::endl;
    std::cout << featureWeights << std::endl;

    // DEBUG //

    /*auto e = computeTrainingEnergy(colorImageFilenames, gtImageFilenames, gtSpImageFilenames, unaryFilenames,
                                   curWeights, properties.pairwiseSigmaSq, numClusters, numClasses, properties.C,
                                   properties, cmap, cmap2);
    std::cout << "Total" << std::endl;
    std::cout << "---> " << e << std::endl;

    return 0;*/



    ThreadPool pool(properties.numThreads);
    std::vector<std::future<SampleResult>> futures;

    float lastTrainingEnergy = std::numeric_limits<float>::max();
    float learningRate = properties.FixedLearningRate.rate;
    if(properties.learningRate == "BoldDriver")
        learningRate = properties.BoldDriverLearningRate.base;
    WeightsVec lastWeights = curWeights;
    WeightsVec lastGradient = curWeights;

    std::ofstream log(properties.log);
    if(!log.is_open())
    {
        std::cerr << "Cannot write to log file \"" << properties.log << "\"" << std::endl;
        return 50;
    }

    // Iterate T times
    for(size_t t = properties.startIter; t < T; ++t)
    {
        WeightsVec sum(numClasses, 0, 0, 0, 0); // All zeros
        float iterationEnergy = 0;
        futures.clear();

        // Shuffle the filenames (so it doesn't always take the same elements)
        std::shuffle(filenames.begin(), filenames.end(), random);

        // Iterate over all images
        for (size_t n = 0; n < N; ++n)
        {
            std::string colorImgFilename, gtImageFilename, unaryFilename;
            colorImgFilename = gtImageFilename = unaryFilename = filenames[n];

            auto&& fut = pool.enqueue(processSample, colorImgFilename, gtImageFilename, unaryFilename, properties, cmap,
                                      numClasses, numClusters, curWeights, featureWeights);
            futures.push_back(std::move(fut));
        }

        for(size_t n = 0; n < futures.size(); ++n)
        {
            auto sampleResult = futures[n].get();
            if(!sampleResult.valid)
            {
                std::cerr << "Sample result was invalid. Cannot continue." << std::endl;
                return 2;
            }

            std::cout << "<<< " << t << "/" << n << " >>>" << std::endl;

            sum += sampleResult.energyDiff;
            iterationEnergy += sampleResult.trainingEnergy;
        }

        // Show current training energy
        iterationEnergy *= properties.C / N;
        iterationEnergy += curWeights.sqNorm() / 2.f;
        std::cout << "Current training energy: " << iterationEnergy << std::endl;

        // Compute learning rate
        if(properties.learningRate == "Diminishing")
            learningRate = properties.DiminishingLearningRate.base / (1 + t / properties.DiminishingLearningRate.T);
        else if(properties.learningRate == "BoldDriver")
        {
            if(t == 0)
                learningRate = properties.BoldDriverLearningRate.base;
            else
            {
                if(iterationEnergy < lastTrainingEnergy)
                    learningRate *= properties.BoldDriverLearningRate.increase;
                else if(iterationEnergy - lastTrainingEnergy > properties.BoldDriverLearningRate.margin)
                {
                    learningRate *= properties.BoldDriverLearningRate.decrease;
                    auto gradient = lastGradient;
                    gradient *= learningRate;
                    curWeights = lastWeights;
                    curWeights -= gradient;
                    t--;
                    continue;
                }
            }
        }

        lastWeights = curWeights;
        lastTrainingEnergy = iterationEnergy;

        log << std::setw(4) << t << "\t" << std::setw(12) << iterationEnergy << "\t" << std::setw(12) << learningRate << std::endl;

        // Update step
        sum *= properties.C / N;
        sum += curWeights;
        lastGradient = sum;
        sum *= learningRate;
        curWeights -= sum;

        // Project onto the feasible set
        curWeights.clampToFeasible();

        if (!curWeights.write(properties.outDir + properties.outFile))
            std::cerr << "Couldn't write weights to file " << properties.outDir + properties.outFile << std::endl;
        curWeights.write(properties.outDir + "iterations/" + std::to_string(t) + ".dat");

        std::cout << "====================" << std::endl;
        std::cout << "Current weights:" << std::endl;
        std::cout << curWeights << std::endl;
    }

    log.close();

    return 0;
}