//
// Created by jan on 01.09.16.
//

#include <BaseProperties.h>
#include <Energy/WeightsVec.h>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include <helper/image_helper.h>
#include <Energy/EnergyFunction.h>
#include <Inference/k-prototypes/Clusterer.h>
#include <Threading/ThreadPool.h>

PROPERTIES_DEFINE(TrainDistMerge,
                  PROP_DEFINE_A(size_t, t, 0, -t)
                  PROP_DEFINE_A(float, learningRate, 1e-7f, -eta)
                  PROP_DEFINE_A(float, T, 1.f, -T)
                  PROP_DEFINE_A(float, C, 1.f, -C)
                  PROP_DEFINE_A(float, pairwiseSigmaSq, 0.05f, -s)
                  PROP_DEFINE_A(size_t, numClusters, 300, -c)
                  PROP_DEFINE_A(std::string, trainingList, "", -l)
                  PROP_DEFINE_A(std::string, weightFile, "", -w)
                  PROP_DEFINE_A(std::string, imgPath, "", -I)
                  PROP_DEFINE_A(std::string, gtPath, "", -G)
                  PROP_DEFINE_A(std::string, gtSpPath, "", -S)
                  PROP_DEFINE_A(std::string, unaryPath, "", -U)
                  PROP_DEFINE_A(std::string, in, "", -i)
                  PROP_DEFINE_A(std::string, out, "", -o)
                  PROP_DEFINE_A(size_t, numThreads, 8, -nt)
                  PROP_DEFINE(bool, useDiminishingStepSize, true)
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
    float energy = 0;
    WeightsVec diff{21ul, false};
    bool valid = false;
};

SampleResult processSample(TrainDistMergeProperties const& properties, std::string filename,
                           helper::image::ColorMap const& cmap, helper::image::ColorMap const& cmap2, size_t numClasses,
                           WeightsVec const& curWeights, WeightsVec const& oneWeights)
{
    SampleResult sampleResult;
    size_t numClusters = properties.numClusters;

    // Load images etc...
    RGBImage rgbImage, groundTruthRGB, groundTruthSpRGB, predictionRGB, predictionSpRGB;
    rgbImage.read(properties.imgPath + filename + ".jpg");
    groundTruthRGB.read(properties.gtPath + filename + ".png");
    groundTruthSpRGB.read(properties.gtSpPath + filename + ".png");
    predictionRGB.read(properties.in + "labeling/" + filename + ".png");
    predictionSpRGB.read(properties.in + "sp/" + filename + ".png");
    if (rgbImage.width() != groundTruthRGB.width() || rgbImage.height() != groundTruthRGB.height() ||
        rgbImage.width() != groundTruthSpRGB.width() || rgbImage.height() != groundTruthSpRGB.height() ||
        rgbImage.width() != predictionRGB.width() || rgbImage.height() != predictionRGB.height() ||
        rgbImage.width() != predictionSpRGB.width() || rgbImage.height() != predictionSpRGB.height())
    {
        std::cerr << "Image " << filename << " and its ground truth and/or prediction don't match." << std::endl;
        return sampleResult;
    }
    CieLabImage cieLabImage = rgbImage.getCieLabImg();
    LabelImage groundTruth = helper::image::decolorize(groundTruthRGB, cmap);
    LabelImage groundTruthSp = helper::image::decolorize(groundTruthSpRGB, cmap2);
    LabelImage prediction = helper::image::decolorize(predictionRGB, cmap);
    LabelImage predictionSp = helper::image::decolorize(predictionSpRGB, cmap2);

    UnaryFile unary(properties.unaryPath + filename + "_prob.dat");
    if(unary.width() != rgbImage.width() || unary.height() != rgbImage.height() || unary.classes() != numClasses)
    {
        std::cerr << "Invalid unary scores " << properties.unaryPath + filename + "_prob.dat" << std::endl;
        return sampleResult;
    }

    EnergyFunction trainingEnergy(unary, curWeights, properties.pairwiseSigmaSq);
    auto clusters = Clusterer::computeClusters(predictionSp, cieLabImage, prediction, numClusters, numClasses);
    sampleResult.energy -= trainingEnergy.giveEnergy(prediction, cieLabImage, predictionSp, clusters);
    auto gtClusters = Clusterer::computeClusters(groundTruthSp, cieLabImage, groundTruth, numClusters, numClasses);
    sampleResult.energy += trainingEnergy.giveEnergy(groundTruth, cieLabImage, groundTruthSp, gtClusters);

    // Compute loss
    float lossFactor = 0;
    for(size_t i = 0; i < groundTruth.pixels(); ++i)
        if(groundTruth.atSite(i) < unary.classes())
            lossFactor++;
    lossFactor = 1e8f / lossFactor;
    float loss = 0;
    for(size_t i = 0; i < groundTruth.pixels(); ++i)
        if (groundTruth.atSite(i) != prediction.atSite(i) && groundTruth.atSite(i) < unary.classes())
            loss += lossFactor;
    sampleResult.energy += loss;

    // Compute energy without weights on the ground truth
    EnergyFunction normalEnergy(unary, oneWeights, properties.pairwiseSigmaSq);
    auto gtEnergy = normalEnergy.giveEnergyByWeight(groundTruth, cieLabImage, groundTruthSp, gtClusters);

    // Compute energy without weights on the prediction
    auto predEnergy = normalEnergy.giveEnergyByWeight(prediction, cieLabImage, predictionSp, clusters);
    // Compute energy difference
    gtEnergy -= predEnergy;

    sampleResult.diff = gtEnergy;
    sampleResult.valid = true;
    return sampleResult;
}

int main(int argc, char* argv[])
{
    // Read properties
    TrainDistMergeProperties properties;
    properties.read("properties/training_dist_merge.info");
    properties.fromCmd(argc, argv);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    size_t const numClasses = 21;
    size_t const numClusters = properties.numClusters;

    helper::image::ColorMap const cmap = helper::image::generateColorMapVOC(std::max(256ul, numClasses));
    helper::image::ColorMap const cmap2 = helper::image::generateColorMap(numClusters);

    WeightsVec oneWeights(numClasses, 1, 1, 1, 1, 1, 1, 1);
    WeightsVec curWeights(numClasses, 0, 0, 0, 0, 0, 0, 0);
    if(!curWeights.read(properties.weightFile) && properties.t != 0)
    {
        std::cerr << "Couldn't read current weights from " << properties.weightFile << std::endl;
        return -1;
    }
    std::cout << "====================" << std::endl;
    std::cout << "Initial weights:" << std::endl;
    std::cout << curWeights << std::endl;
    std::cout << "====================" << std::endl;

    // Read in list of files
    std::vector<std::string> list = readFileNames(properties.trainingList);
    if(list.empty())
    {
        std::cerr << "File list \"" << properties.trainingList << "\" is empty." << std::endl;
        return -2;
    }

    // Iterate over all predictions
    size_t N = list.size();
    size_t t = properties.t;
    WeightsVec sum(numClasses, false);
    ThreadPool pool(properties.numThreads);
    std::vector<std::future<SampleResult>> futures;
    // Iterate over all images
    for (size_t n = 0; n < N; ++n)
    {
        auto&& fut = pool.enqueue(processSample, properties, list[n], cmap, cmap2, numClasses, curWeights, oneWeights);
        futures.push_back(std::move(fut));
    }

    float trainingEnergy = 0;
    for(size_t n = 0; n < futures.size(); ++n)
    {
        auto sampleResult = futures[n].get();
        if(!sampleResult.valid)
        {
            std::cerr << "Sample result was invalid. Cannot continue." << std::endl;
            return 2;
        }

        sum += sampleResult.diff;
        trainingEnergy += sampleResult.energy;

        std::cout << "<<< " << t << "/" << n << " >>>" << std::endl;
    }

    // Show current training energy
    trainingEnergy *= properties.C / N;
    trainingEnergy += curWeights.sqNorm() / 2.f;
    std::cout << "Current training energy: " << trainingEnergy << std::endl;
    boost::filesystem::path energyFilePath(properties.out);
    energyFilePath.remove_filename();
    std::ofstream out(energyFilePath.string() + "/training_energy.txt", std::ios::out | std::ios::app);
    if(out.is_open())
    {
        out.precision(std::numeric_limits<float>::max_digits10);
        out << std::setw(3) << properties.t << ": ";
        out << trainingEnergy << std::endl;
        out.close();
    }
    else
        std::cerr << "Couldn't write current training energy to file " << energyFilePath << std::endl;

    // Update step
    sum *= properties.C / N;
    sum += curWeights;

    std::cout << "Gradient: " << sum << std::endl;

    float stepSize = properties.learningRate;
    if (properties.useDiminishingStepSize)
        stepSize /= 1 + t / properties.T;
    std::cout << "Step size: " << stepSize << std::endl;

    sum *= stepSize;
    curWeights -= sum;

    if(!curWeights.write(properties.out))
        std::cerr << "Couldn't write weights to file " << properties.out << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << curWeights << std::endl;
    std::cout << "====================" << std::endl;

    curWeights.write(energyFilePath.string() + "/weights/" + std::to_string(properties.t + 1) + ".dat");

    return 0;
}