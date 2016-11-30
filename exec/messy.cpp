#include <iostream>
#include <Energy/UnaryFile.h>
#include <boost/filesystem.hpp>
#include <Energy/WeightsVec.h>
#include <map>
#include <set>
#include <Inference/TRW_S_Optimizer/TRW_S_Optimizer.h>
#include <Energy/EnergyFunction.h>
#include <Energy/feature_weights.h>
#include <Inference/k-prototypes/Clusterer.h>
#include <helper/image_helper.h>
#include <Inference/GraphOptimizer/GraphOptimizer.h>
#include "Timer.h"

float loss(int y, int gt)
{
    if(y == gt)
        return 0.f;
    else
        return 1.f;
}

float energy(float x, int y, float w)
{
    return (x - y) * (x - y) + w * y * y;
}

int predict(float x, float w, int const maxLabel, int gt = -1)
{
    float minEnergy = energy(x, 0, w);
    int minLabel = 0;
    if (gt > 0)
        minEnergy -= loss(0, gt);
    for(int l = 1; l <= maxLabel; ++l)
    {
        float e = energy(x, l, w);
        if (gt >= 0 && l != gt)
            e -= loss(l, gt);
        if(e < minEnergy)
        {
            minEnergy = e;
            minLabel = l;
        }
    }
    return minLabel;
}

float
trainingEnergy(std::vector<float> const& x, std::vector<int> const& gt, std::vector<int> const& pred, float w, float C,
               float* lossVal = nullptr)
{
    if(lossVal != nullptr)
        *lossVal = 0;
    float e = w * w / 2.f;
    float sum = 0;
    for(size_t n = 0; n < x.size(); ++n)
    {
        sum += loss(pred[n], gt[n]) + energy(x[n], gt[n], w) - energy(x[n], pred[n], w);
        if(lossVal != nullptr)
        {
            int y = predict(x[n], w, x.size());
            *lossVal += loss(y, gt[n]);
        }
    }
    e +=  C / x.size() * sum;
    e += 10;
    return e;
}


int main()
{
    size_t numLabels = 21;
    UnaryFile unary("data/2007_000129_prob.dat");
    WeightsVec weights(numLabels, 100, 1000000, 10, 0);
    Matrix5f featureWeights = readFeatureWeights("out/featureWeights.txt");
    featureWeights = featureWeights.inverse();
    EnergyFunction energy(unary, weights, 0.5f, featureWeights);

    RGBImage rgb;
    rgb.read("data/2007_000129.jpg");
    auto cieLab = rgb.getCieLabImg();

    size_t numClusters = 200;
    Clusterer<EnergyFunction> clusterer(energy);
    auto maxLabeling = LabelImage(unary.width(), unary.height());//unary.maxLabeling();
    clusterer.run(numClusters, numLabels, cieLab, maxLabeling);
    auto const& sp = clusterer.clustership();

    TRW_S_Optimizer<EnergyFunction> trwsOptimizer(energy);
    trwsOptimizer.run(cieLab, sp, numClusters);
    auto const& labeling_trws = trwsOptimizer.labeling();

    GraphOptimizer<EnergyFunction> graphOptimizer(energy);
    graphOptimizer.run(cieLab, sp, numClusters);
    auto const& labeling_graph = graphOptimizer.labeling();

    auto cmap = helper::image::generateColorMapVOC(numLabels);
    auto labelImg_trws = helper::image::colorize(labeling_trws, cmap);
    auto labelImg_graph = helper::image::colorize(labeling_graph, cmap);
    cv::Mat cvLabeling_trws = static_cast<cv::Mat>(labelImg_trws);
    cv::Mat cvLabeling_graph = static_cast<cv::Mat>(labelImg_graph);
    cv::imshow("TRW_S", cvLabeling_trws);
    cv::imshow("Graph", cvLabeling_graph);
    cv::waitKey();

#if 0
    std::vector<float> x = {0.f, 1.6f, 0.95f, 1.55f};
    std::vector<int> gt = {0, 1, 1, 1};
    std::vector<int> y = {0, 0, 0, 0};
    std::vector<int> predictions = {0, 0, 0, 0};
    int const maxLabel = 2;

    size_t const T = 5000;
    size_t const N = x.size();
    float const C = 1.f;
    float const eta = 0.3f;

    float wCur = -10.f;
    std::vector<float> trainingEnergies;
    for (size_t t = 0; t < T; ++t)
    {
        std::cout << "Iteration " << t << std::endl;
        float sum = 0;
        for(size_t n = 0; n < N; ++n)
        {
            int pred = predict(x[n], wCur, maxLabel, gt[n]);
            y[n] = pred;
            float predEnergy = energy(x[n], pred, 1);
            float gtEnergy = energy(x[n], gt[n], 1);
            sum += gtEnergy - predEnergy;

            predictions[n] = predict(x[n], wCur, maxLabel);
            std::cout << n << ": " << predictions[n] << "/" << gt[n] << " | ";
        }
        std::cout << std::endl;

        float loss = 0;
        float e = trainingEnergy(x, gt, y, wCur, C, &loss);
        trainingEnergies.push_back(e);
        std::cout << "Training energy = " << e << " | Loss = " << loss << std::endl;

        float p = wCur + C/N * sum;
        wCur -= eta / (t+1) * p;
        std::cout << "New wCur = " << wCur << std::endl;
    }

    std::cout << "-----------" << std::endl;
    for(size_t i = 0; i < trainingEnergies.size(); ++i)
    {
        std::cout << i << "\t" << trainingEnergies[i] << std::endl;
    }
#endif

    return 0;
}