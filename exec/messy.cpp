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
#include <Timer.h>

int main()
{
    size_t numLabels = 21;
    size_t numClusters = 200;
    UnaryFile unary("data/2007_000129_prob.dat");
    WeightsVec weights(numLabels, 100, 10, 10, 10);
    weights.read("out/weights_multi_large.dat");
    Matrix5f featureWeights = readFeatureWeights("out/featureWeights.txt");
    featureWeights = featureWeights.inverse();
    EnergyFunction energy(unary, weights, 0.5f, featureWeights);
    auto cmap = helper::image::generateColorMapVOC(numLabels);
    auto cmap2 = helper::image::generateColorMap(numClusters);

    RGBImage rgb, gtRgb;
    rgb.read("data/2007_000129.jpg");
    auto cieLab = rgb.getCieLabImg();
    gtRgb.read("data/2007_000129.png");
    LabelImage gt;
    gt = helper::image::decolorize(gtRgb, cmap);

    Timer t(true);
    Clusterer<EnergyFunction> clusterer(energy);
    size_t numIter= clusterer.run(numClusters, numLabels, cieLab, gt);
    t.pause();

    std::cout << numIter << " iterations (" << t.elapsed() << ")" << std::endl;

    auto clusters = helper::image::colorize(clusterer.clustership(), cmap2);
    cv::Mat clusterMat = static_cast<cv::Mat>(clusters);

    cv::imshow("RGB", static_cast<cv::Mat>(rgb));
    cv::imshow("GT", static_cast<cv::Mat>(gtRgb));
    cv::imshow("Clusters", clusterMat);

    cv::waitKey();

    clusters.write("out/2007_000129_clustering_new.png");


    return 0;
}