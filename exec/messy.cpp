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

int smoothCostFun(int s1, int s2, int l1, int l2)
{
    return l1 != l2 ? 1 : 0;
}

LabelImage tryAlphaBeta(Image<float, 3> const& img, UnaryFile const& unary, float lambda)
{
    size_t numPx = img.pixels();
    size_t numNodes = numPx;

    // Setup graph
    GCoptimizationGeneralGraph graph(numNodes, unary.classes());
    for (size_t i = 0; i < numPx; ++i)
    {
        auto coords = helper::coord::siteTo2DCoordinate(i, img.width());

        // Set up pixel neighbor connections
        decltype(coords) coordsR = {coords.x() + 1, coords.y()};
        decltype(coords) coordsD = {coords.x(), coords.y() + 1};
        if (coordsR.x() < img.width())
        {
            size_t siteR = helper::coord::coordinateToSite(coordsR.x(), coordsR.y(), img.width());
            graph.setNeighbors(i, siteR, lambda);
        }
        if (coordsD.y() < img.height())
        {
            size_t siteD = helper::coord::coordinateToSite(coordsD.x(), coordsD.y(), img.width());
            graph.setNeighbors(i, siteD, lambda);
        }
        // Set up unary cost
        for (Label l = 0; l < unary.classes(); ++l)
            graph.setDataCost(i, l, -unary.at(coords.x(), coords.y(), l));
    }

    // Set up pairwise cost
    graph.setSmoothCost(smoothCostFun);

    // Do alpha-beta-swap
    try
    {
        graph.swap();
    }
    catch(GCException e)
    {
        e.Report();
    }

    // Copy over result
    LabelImage labeling(img.width(), img.height());
    for (size_t i = 0; i < numPx; ++i)
        labeling.atSite(i) = graph.whatLabel(i);

    return labeling;
}

LabelImage tryTrwS(Image<float, 3> const& img, UnaryFile const& unary, float lambda)
{
    size_t numPx = img.pixels();
    size_t numNodes = numPx;
    size_t numClasses = unary.classes();

    // Set up the graph
    TypePotts::GlobalSize globalSize(numClasses);
    MRFEnergy<TypePotts> mrfEnergy(globalSize);

    std::vector<MRFEnergy<TypePotts>::NodeId> nodeIds;
    nodeIds.reserve(numNodes);

    // Unaries
    for (size_t i = 0; i < numNodes; ++i)
    {
        // Unary confidences
        std::vector<TypePotts::REAL> confidences(numClasses, 0.f);
        if (i < numPx) // Only nodes that represent pixels have unaries.
            for (size_t l = 0; l < numClasses; ++l)
                confidences[l] = -unary.atSite(i, l);
        auto id = mrfEnergy.AddNode(TypePotts::LocalSize(), TypePotts::NodeData(confidences.data()));
        nodeIds.push_back(id);
    }

    // Pairwise
    for (size_t i = 0; i < numPx; ++i)
    {
        auto coords = helper::coord::siteTo2DCoordinate(i, img.width());

        // Set up pixel neighbor connections
        decltype(coords) coordsR = {coords.x() + 1, coords.y()};
        decltype(coords) coordsD = {coords.x(), coords.y() + 1};
        if (coordsR.x() < img.width())
        {
            size_t siteR = helper::coord::coordinateToSite(coordsR.x(), coordsR.y(), img.width());
            TypePotts::EdgeData edgeData(lambda);
            mrfEnergy.AddEdge(nodeIds[i], nodeIds[siteR], edgeData);
        }
        if (coordsD.y() < img.height())
        {
            size_t siteD = helper::coord::coordinateToSite(coordsD.x(), coordsD.y(), img.width());
            TypePotts::EdgeData edgeData(lambda);
            mrfEnergy.AddEdge(nodeIds[i], nodeIds[siteD], edgeData);
        }
    }

    // Do the actual minimization
    MRFEnergy<TypePotts>::Options options;
    options.m_eps = 0.0f;
    options.m_printIter=1000000;
    options.m_printMinIter=1000000;
    MRFEnergy<TypePotts>::REAL lowerBound = 0, energy = 0;
    mrfEnergy.Minimize_TRW_S(options, lowerBound, energy);

    // Copy over result
    LabelImage labeling(img.width(), img.height());
    for (size_t i = 0; i < numPx; ++i)
        labeling.atSite(i) = mrfEnergy.GetSolution(nodeIds[i]);

    return labeling;
}


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

    RGBImage rgb, gtRgb;
    rgb.read("data/2007_000129.jpg");
    auto cieLab = rgb.getCieLabImg();
    gtRgb.read("data/2007_000129.png");
    LabelImage gt;
    gt = helper::image::decolorize(gtRgb, cmap);

    size_t const numIter = 10;
    Timer::milliseconds alpha_beta_ms = Timer::milliseconds::zero();
    LabelImage labeling_graph;
    for(size_t i = 0; i < numIter; ++i)
    {
        Timer t(true);
        labeling_graph = tryAlphaBeta(cieLab, unary, 5);
        t.pause();
        alpha_beta_ms += t.elapsed();
    }
    alpha_beta_ms /= numIter;
    std::cout << "Alpha-Beta-Swap: " << alpha_beta_ms << std::endl;

    Timer::milliseconds trws_ms = Timer::milliseconds::zero();
    LabelImage labeling_trws;
    for(size_t i = 0; i < numIter; ++i)
    {
        Timer t(true);
        labeling_trws = tryTrwS(cieLab, unary, 5);
        t.pause();
        trws_ms += t.elapsed();
    }
    trws_ms /= numIter;
    std::cout << "TRW-S: " << trws_ms << std::endl;

    auto labelImg_trws = helper::image::colorize(labeling_trws, cmap);
    auto labelImg_graph = helper::image::colorize(labeling_graph, cmap);
    cv::Mat cvLabeling_trws = static_cast<cv::Mat>(labelImg_trws);
    cv::Mat cvLabeling_graph = static_cast<cv::Mat>(labelImg_graph);
    cv::imwrite("trws.png", cvLabeling_trws);
    cv::imwrite("alphabeta.png", cvLabeling_graph);
    cv::imshow("TRW_S", cvLabeling_trws);
    cv::imshow("Graph", cvLabeling_graph);
    cv::waitKey();

#if 0
    Timer t(true);
    InferenceIterator<EnergyFunction, TRW_S_Optimizer> inference_trws(energy, numClusters, numLabels, cieLab);
    auto result = inference_trws.run();
    auto labeling_trws = result.labeling;
    t.pause();
    std::cout << "TRWS took " << t.elapsed<Timer::seconds>() << std::endl;

    t.reset(true);
    InferenceIterator<EnergyFunction, GraphOptimizer> inference_graph(energy, numClusters, numLabels, cieLab);
    result = inference_graph.run();
    auto labeling_graph = result.labeling;
    t.pause();
    std::cout << "Alpha-Beta-Swap took " << t.elapsed<Timer::seconds>() << std::endl;

    // Compute accuracy
    size_t correct_trws = 0, correct_graph = 0;
    for(size_t i = 0; i < gt.pixels(); ++i)
    {
        if(labeling_trws.atSite(i) == gt.atSite(i))
            correct_trws++;
        if(labeling_graph.atSite(i) == gt.atSite(i))
            correct_graph++;
    }
    std::cout << "Accuracy TRWS: " << correct_trws / (float) gt.pixels() << std::endl;
    std::cout << "Accuracy Graph: " << correct_graph / (float) gt.pixels() << std::endl;

    auto labelImg_trws = helper::image::colorize(labeling_trws, cmap);
    auto labelImg_graph = helper::image::colorize(labeling_graph, cmap);
    cv::Mat cvLabeling_trws = static_cast<cv::Mat>(labelImg_trws);
    cv::Mat cvLabeling_graph = static_cast<cv::Mat>(labelImg_graph);
    cv::imwrite("trws.png", cvLabeling_trws);
    cv::imwrite("alphabeta.png", cvLabeling_graph);
    cv::imshow("TRW_S", cvLabeling_trws);
    cv::imshow("Graph", cvLabeling_graph);
    cv::waitKey();
#endif

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