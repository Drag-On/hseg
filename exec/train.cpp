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
                  PROP_DEFINE(std::string, imageListFile, "")
                  PROP_DEFINE(std::string, groundTruthListFile, "")
                  PROP_DEFINE(std::string, groundTruthSpListFile, "")
                  PROP_DEFINE(std::string, unaryListFile, "")
                  PROP_DEFINE(std::string, imageBasePath, "")
                  PROP_DEFINE(std::string, groundTruthBasePath, "")
                  PROP_DEFINE(std::string, groundTruthSpBasePath, "")
                  PROP_DEFINE(std::string, unaryBasePath, "")
                  PROP_DEFINE(std::string, imageExtension, ".jpg")
                  PROP_DEFINE(std::string, gtExtension, ".png")
                  PROP_DEFINE(std::string, in, "")
                  PROP_DEFINE(std::string, out, "out/weights.dat")
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

struct TrainingEnergy
{
    float total = 0;
    float pred = 0;
    float gt = 0;
    float loss = 0;
    float regularizer = 0;

    TrainingEnergy& operator+=(TrainingEnergy const& other)
    {
        total += other.total;
        pred += other.pred;
        gt += other.gt;
        loss += other.loss;
        regularizer += other.regularizer;
        return *this;
    }
};

std::ostream& operator<<(std::ostream& stream, TrainingEnergy const& e)
{
    stream << "Total: " << e.total << " ( pred: " << e.pred << ", gt: " << e.gt << ", loss: " << e.loss << ", regularizer: " << e.regularizer << ")";
    return stream;
}

TrainingEnergy
computeTrainingSampleEnergy(WeightsVec const& weights, UnaryFile const& unary, CieLabImage const& cieLabImg,
                            LabelImage const& maxLabeling, LabelImage const& maxSp, LabelImage const& gtImg,
                            LabelImage const& gtSpImg, float pairwiseSigmaSq, size_t numClusters, size_t numClasses,
                            float C, size_t N, Matrix5f const& featureWeights)
{
    TrainingEnergy e;
    EnergyFunction trainingEnergy(unary, weights, pairwiseSigmaSq, featureWeights);

    float cOverN = C / N;

    // Energy on prediction
    auto clusters = Clusterer::computeClusters(maxSp, cieLabImg, maxLabeling, numClusters, numClasses, trainingEnergy);
    e.pred = cOverN * trainingEnergy.giveEnergy(maxLabeling, cieLabImg, maxSp, clusters);
    e.total -= e.pred;

    // Energy on ground truth
    auto gtClusters = Clusterer::computeClusters(gtSpImg, cieLabImg, gtImg, numClusters, numClasses, trainingEnergy);
    e.gt = cOverN * trainingEnergy.giveEnergy(gtImg, cieLabImg, gtSpImg, gtClusters);
    e.total += e.gt;

    // Loss
    float lossFactor = 0;
    for(size_t i = 0; i < gtImg.pixels(); ++i)
        if(gtImg.atSite(i) < unary.classes())
            lossFactor++;
    lossFactor = 1e8f / lossFactor;
    for(size_t i = 0; i < gtImg.pixels(); ++i)
        if (gtImg.atSite(i) != maxLabeling.atSite(i) && gtImg.atSite(i) < unary.classes())
            e.loss += lossFactor;
    e.loss *= cOverN;
    e.total += e.loss;

    return e;
}

TrainingEnergy computeTrainingEnergy(std::vector<std::string> const& clrImgs, std::vector<std::string> const& gtImgs,
                                     std::vector<std::string> const& gtSpImgs, std::vector<std::string> const& unaries,
                                     WeightsVec const& weights, float pairwiseSigmaSq, size_t numClusters,
                                     size_t numClasses, float C, TrainProperties const& props,
                                     helper::image::ColorMap const cmap, helper::image::ColorMap const& cmap2,
                                     Matrix5f const& featureWeights)
{
    TrainingEnergy totalE;
    size_t N = clrImgs.size();

    ThreadPool pool(props.numThreads);
    std::vector<std::future<TrainingEnergy>> futures;

    auto doOneSample = [&](size_t n) -> TrainingEnergy
    {
        // Load images etc...
        RGBImage rgbImage, groundTruthRGB, groundTruthSpRGB;
        rgbImage.read(props.imageBasePath + clrImgs[n] + props.imageExtension);
        groundTruthRGB.read(props.groundTruthBasePath + gtImgs[n] + props.gtExtension);
        groundTruthSpRGB.read(props.groundTruthSpBasePath + gtSpImgs[n] + props.gtExtension);

        CieLabImage cieLabImage = rgbImage.getCieLabImg();
        LabelImage groundTruth = helper::image::decolorize(groundTruthRGB, cmap);
        LabelImage groundTruthSp = helper::image::decolorize(groundTruthSpRGB, cmap2);

        UnaryFile unary(props.unaryBasePath + unaries[n] + "_prob.dat");

        // Predict with loss-augmented energy
        LossAugmentedEnergyFunction energy(unary, weights, pairwiseSigmaSq, featureWeights, groundTruth);
        InferenceIterator inference(energy, numClusters, numClasses, cieLabImage);
        InferenceResult result = inference.run(2);

        // Compute training sample energy
        TrainingEnergy e = computeTrainingSampleEnergy(weights, unary, cieLabImage, result.labeling, result.superpixels,
                                                       groundTruth, groundTruthSp, pairwiseSigmaSq, numClusters,
                                                       numClasses, C, N, featureWeights);
        return e;
    };

    for (size_t n = 0; n < N; ++n)
    {
        auto&& fut = pool.enqueue(doOneSample, n);
        futures.push_back(std::move(fut));
    }

    for(size_t n = 0; n < N; ++n)
    {
        TrainingEnergy e = futures[n].get();
        std::cout << "Sample " << n << std::endl;
        std::cout << "---> " << e << std::endl;
        totalE += e;
    }

    // Add regularizer
    totalE.regularizer = weights.sqNorm() / 2.f;
    return totalE;
}

struct SampleResult
{
    WeightsVec energyDiff{21ul, false};
    float trainingEnergy = 0;
    bool valid = false;
};

SampleResult processSample(std::string const& colorImgFilename, std::string const& gtImageFilename,
                           std::string const& gtSpImageFilename, std::string const& unaryFilename,
                           TrainProperties const& properties, helper::image::ColorMap const& cmap,
                           helper::image::ColorMap const& cmap2, size_t numClasses, size_t numClusters,
                           WeightsVec const& curWeights, WeightsVec const& oneWeights, Matrix5f const& featureWeights)
{
    SampleResult sampleResult;

    // Load images etc...
    RGBImage rgbImage, groundTruthRGB, groundTruthSpRGB;
    rgbImage.read(properties.imageBasePath + colorImgFilename + properties.imageExtension);
    groundTruthRGB.read(properties.groundTruthBasePath + gtImageFilename + properties.gtExtension);
    groundTruthSpRGB.read(properties.groundTruthSpBasePath + gtSpImageFilename + properties.gtExtension);
    if (rgbImage.width() != groundTruthRGB.width() || rgbImage.height() != groundTruthRGB.height() ||
        rgbImage.width() != groundTruthSpRGB.width() || rgbImage.height() != groundTruthSpRGB.height())
    {
        std::cerr << "Image " << colorImgFilename << " and its ground truth don't match." << std::endl;
        return sampleResult;
    }
    CieLabImage cieLabImage = rgbImage.getCieLabImg();
    LabelImage groundTruth = helper::image::decolorize(groundTruthRGB, cmap);
    LabelImage groundTruthSp = helper::image::decolorize(groundTruthSpRGB, cmap2);

    UnaryFile unary(properties.unaryBasePath + unaryFilename + "_prob.dat");
    if(unary.width() != rgbImage.width() || unary.height() != rgbImage.height() || unary.classes() != numClasses)
    {
        std::cerr << "Invalid unary scores " << unaryFilename << std::endl;
        return sampleResult;
    }

    // Predict with loss-augmented energy
    LossAugmentedEnergyFunction energy(unary, curWeights, properties.pairwiseSigmaSq, featureWeights, groundTruth);
    InferenceIterator inference(energy, properties.numClusters, numClasses, cieLabImage);
    InferenceResult result = inference.run(2);

    /*cv::imshow("sp", static_cast<cv::Mat>(helper::image::colorize(result.superpixels, cmap2)));
    cv::imshow("prediction", static_cast<cv::Mat>(helper::image::colorize(result.labeling, cmap)));
    cv::imshow("unary", static_cast<cv::Mat>(helper::image::colorize(unary.maxLabeling(), cmap)));
    cv::imshow("gt", static_cast<cv::Mat>(groundTruthRGB));
    cv::imshow("gt sp", static_cast<cv::Mat>(groundTruthSpRGB));
    cv::waitKey();*/

    EnergyFunction trainingEnergy(unary, curWeights, properties.pairwiseSigmaSq, featureWeights);
    auto clusters = Clusterer::computeClusters(result.superpixels, cieLabImage, result.labeling, numClusters,
                                               numClasses, trainingEnergy);
    sampleResult.trainingEnergy -= trainingEnergy.giveEnergy(result.labeling, cieLabImage, result.superpixels, clusters);
    auto gtClusters = Clusterer::computeClusters(groundTruthSp, cieLabImage, groundTruth, numClusters, numClasses,
                                                 trainingEnergy);
    sampleResult.trainingEnergy += trainingEnergy.giveEnergy(groundTruth, cieLabImage, groundTruthSp, gtClusters);

    // Compute loss
    float lossFactor = 0;
    for(size_t i = 0; i < groundTruth.pixels(); ++i)
        if(groundTruth.atSite(i) < unary.classes())
            lossFactor++;
    lossFactor = 1e8f / lossFactor;
    float loss = 0;
    for(size_t i = 0; i < groundTruth.pixels(); ++i)
        if (groundTruth.atSite(i) != result.labeling.atSite(i) && groundTruth.atSite(i) < unary.classes())
            loss += lossFactor;
    sampleResult.trainingEnergy += loss;

    // Compute energy without weights on the ground truth
    EnergyFunction normalEnergy(unary, oneWeights, properties.pairwiseSigmaSq, featureWeights);
    auto gtEnergy = normalEnergy.giveEnergyByWeight(groundTruth, cieLabImage, groundTruthSp, gtClusters);

    // Compute energy without weights on the prediction
    auto predEnergy = normalEnergy.giveEnergyByWeight(result.labeling, cieLabImage, result.superpixels,
                                                      clusters);
    // Compute energy difference
    gtEnergy -= predEnergy;

    sampleResult.energyDiff = gtEnergy;
    sampleResult.valid = true;
    return sampleResult;
}

int main()
{
    // Read properties
    TrainProperties properties;
    properties.read("properties/training.info");
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    size_t const numClasses = 21;
    size_t const numClusters = properties.numClusters;
    helper::image::ColorMap const cmap = helper::image::generateColorMapVOC(std::max(256ul, numClasses));
    helper::image::ColorMap const cmap2 = helper::image::generateColorMap(properties.numClusters);
    WeightsVec curWeights(numClasses, 0, 0, 0, 0);
    if(!curWeights.read(properties.in))
        std::cout << "Couldn't read in weights to start from. Using default weights." << std::endl;
    WeightsVec oneWeights(numClasses, 1, 1, 1, 1);

    std::cout << "====================" << std::endl;
    std::cout << "Initial weights:" << std::endl;
    std::cout << curWeights << std::endl;

    // Load filenames of all images
    std::vector<std::string> colorImageFilenames = readFileNames(properties.imageListFile);
    std::vector<std::string> gtImageFilenames = readFileNames(properties.groundTruthListFile);
    std::vector<std::string> gtSpImageFilenames = readFileNames(properties.groundTruthSpListFile);
    std::vector<std::string> unaryFilenames = readFileNames(properties.unaryListFile);
    if (colorImageFilenames.size() != gtImageFilenames.size() || gtImageFilenames.size() != unaryFilenames.size() ||
        gtImageFilenames.size() != gtSpImageFilenames.size())
    {
        std::cerr << "File lists don't match up!" << std::endl;
        return 1;
    }
    size_t T = properties.numIter;
    size_t N = colorImageFilenames.size();

    Matrix5f featureWeights = readFeatureWeights(properties.featureWeightFile);
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
    for(size_t t = 0; t < T; ++t)
    {
        WeightsVec sum(numClasses, 0, 0, 0, 0); // All zeros
        float iterationEnergy = 0;
        futures.clear();

        // Iterate over all images
        for (size_t n = 0; n < N; ++n)
        {
            auto colorImgFilename = colorImageFilenames[n];
            auto gtImageFilename = gtImageFilenames[n];
            auto gtSpImageFilename = gtSpImageFilenames[n];
            auto unaryFilename = unaryFilenames[n];

            auto&& fut = pool.enqueue(processSample, colorImgFilename, gtImageFilename, gtSpImageFilename,
                                      unaryFilename, properties, cmap, cmap2, numClasses, numClusters, curWeights,
                                      oneWeights, featureWeights);
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

        if (!curWeights.write(properties.out))
            std::cerr << "Couldn't write weights to file " << properties.out << std::endl;

        std::cout << "====================" << std::endl;
        std::cout << "Current weights:" << std::endl;
        std::cout << curWeights << std::endl;
    }

    log.close();

    return 0;
}