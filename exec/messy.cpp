#include <iostream>
#include <array>
#include <vector>
#include <Timer.h>
#include <numeric>
#include <iomanip>
#include <typedefs.h>

using Feature = float;
using Image = std::array<Feature, 2>;
//using Label = int;
using Labeling = std::array<Label, 2>;
using Weight = float;
using WeightVec = std::array<Weight, 6 + 9 * 2>; // 6 unary weigths + 9 pairwise combinations a 2 weights
using Cost = float;

WeightVec const zeroWeights = {};

WeightVec operator+(WeightVec w1, WeightVec const& w2)
{
    for(size_t i = 0; i < w1.size(); ++i)
        w1[i] += w2[i];
    return w1;
}

WeightVec operator-(WeightVec w1, WeightVec const& w2)
{
    for(size_t i = 0; i < w1.size(); ++i)
        w1[i] -= w2[i];
    return w1;
}

WeightVec operator*(WeightVec w, Cost c)
{
    for(size_t i = 0; i < w.size(); ++i)
        w[i] *= c;
    return w;
}

Cost loss(Labeling y, Labeling gt)
{
    Cost loss = 0;
    for (size_t i = 0; i < y.size(); ++i)
    {
        if (y[i] != gt[i])
            loss += 1.f;
    }
    return loss;
}

Cost energy(Image x, Labeling y, WeightVec w)
{
    // Sum_i (x_i - y_i)^2 + w_{y_i} * y_i^2 + Sum_ij w_{y_i,y_j}^T * x
    return (x[0] - y[0]) * (x[0] - y[0]) + w[y[0]] * y[0] * y[0] // First unary
           + (x[1] - y[1]) * (x[1] - y[1]) + w[3 + y[1]] * y[1] * y[1] // Second unary
           + w[6 + y[0] + y[1] * 3 + 0 * 3 * 2] * x[0] + w[6 + y[0] + y[1] * 3 + 1 * 3 * 2] * x[1]; // Pairwise
}

WeightVec energy_by_weight(Image x, Labeling y)
{
    WeightVec w = zeroWeights;
    w[y[0]] = y[0] * y[0];
    w[3 + y[1]] = y[1] * y[1];
    w[6 + y[0] + y[1] * 3 + 0 * 3 * 2] = x[0];
    w[6 + y[0] + y[1] * 3 + 1 * 3 * 2] = x[1];
    return w;
}

Labeling predict(Image x, WeightVec w, Label const maxLabel)
{
    Labeling minLabeling = {0, 0};
    Cost minEnergy = energy(x, minLabeling, w);

    for (Label l1 = 0; l1 <= maxLabel; ++l1)
    {
        for (Label l2 = 0; l2 <= maxLabel; ++l2)
        {
            Labeling labeling{l1, l2};
            Cost e = energy(x, labeling, w);
            if (e < minEnergy)
            {
                minEnergy = e;
                minLabeling = labeling;
            }
        }
    }
    return minLabeling;
}

Labeling predict_loss_augmented(Image x, WeightVec w, Label const maxLabel, Labeling gt)
{
    Labeling minLabeling = {0, 0};
    Cost minEnergy = energy(x, minLabeling, w);
    minEnergy -= loss(minLabeling, gt);

    for (Label l1 = 0; l1 <= maxLabel; ++l1)
    {
        for (Label l2 = 0; l2 <= maxLabel; ++l2)
        {
            Labeling labeling{l1, l2};
            Cost e = energy(x, labeling, w);
            e -= loss(labeling, gt);
            if (e < minEnergy)
            {
                minEnergy = e;
                minLabeling = labeling;
            }
        }
    }
    return minLabeling;
}

Cost
training_energy(std::vector<Image> const& x, std::vector<Labeling> const& gt, std::vector<Labeling> const& pred,
                WeightVec w, Cost C)
{
    // Regularizer cost
    Cost e = std::accumulate(w.begin(), w.end(), 0, [](Cost w, Cost a) { return w + a * a; });

    // Upper bound
    Cost sum = 0;
    for (size_t n = 0; n < x.size(); ++n)
    {
        sum += loss(pred[n], gt[n]) + energy(x[n], gt[n], w) - energy(x[n], pred[n], w);
    }
    e += C / x.size() * sum;
    return e;
}

Cost training_loss(std::vector<Image> const& x, std::vector<Labeling> const& gt, WeightVec w, Cost C, Label maxLabel)
{
    Cost lossVal = 0;
    for (size_t n = 0; n < x.size(); ++n)
    {
        Labeling pred = predict(x[n], w, maxLabel);
        lossVal += loss(pred, gt[n]);
    }
    lossVal *= C / x.size();
    return lossVal;
}

int main()
{
    std::vector<Image> x = {{0.f,   0.2f},
                            {0.7f,  0.1f},
                            {0.9f,  1.6f},
                            {0.45f, 0.95f},
                            {1.55f, 1.6f}};
    std::vector<Labeling> gt = {{0, 0},
                                {1, 0},
                                {1, 1},
                                {1, 1},
                                {1, 1}};
    std::vector<Labeling> y = {{0, 0},
                               {0, 0},
                               {0, 0},
                               {0, 0},
                               {0, 0}};
    std::vector<Labeling> predictions = {{0, 0},
                                         {0, 0},
                                         {0, 0},
                                         {0, 0},
                                         {0, 0}};
    int const maxLabel = 2;

    size_t const T = 500;
    size_t const N = x.size();
    Cost const C = 1.f;
    Cost const eta = 0.01f;

    WeightVec wCur = zeroWeights;
    std::vector<Cost> trainingEnergies;
    std::vector<Cost> trainingLoss;
    for (size_t t = 0; t < T; ++t)
    {
        std::cout << "Iteration " << t << std::endl;
        WeightVec sum = zeroWeights;
        for (size_t n = 0; n < N; ++n)
        {
            Labeling pred = predict_loss_augmented(x[n], wCur, maxLabel, gt[n]);
            y[n] = pred;
            WeightVec predEnergy = energy_by_weight(x[n], pred);
            WeightVec gtEnergy = energy_by_weight(x[n], gt[n]);
            sum = sum + (gtEnergy - predEnergy);

            predictions[n] = predict(x[n], wCur, maxLabel);
            std::cout << n << ": " << predictions[n][0] << "," << predictions[n][1] << "/" << gt[n][0] << "," << gt[n][1] << " | ";
        }
        std::cout << std::endl;

        Cost e = training_energy(x, gt, y, wCur, C);
        Cost lossValue = training_loss(x, gt, wCur, C, maxLabel);
        trainingEnergies.push_back(e);
        trainingLoss.push_back(lossValue);
        std::cout << "Training upperBound = " << e << " | Loss = " << lossValue << std::endl;

        WeightVec p = wCur + sum * (C / N);
        wCur = wCur - p * (eta / (t + 1));
        std::cout << "New wCur = ";
        for (auto w : wCur)
            std::cout << w << ", ";
        std::cout << std::endl;
    }

    std::cout << "-----------" << std::endl;
    for (size_t i = 0; i < trainingEnergies.size(); ++i)
    {
        std::cout << i << "\t" << trainingEnergies[i] << "\t" << trainingLoss[i] << std::endl;
    }

    // Formatted output of the weights
    std::cout << "-----------" << std::endl;
    std::cout << std::fixed;
    std::cout.precision(2);

    // Weight for first pixel
    std::cout << "w1" << std::endl;
    for(Label l = 0; l <= maxLabel; ++l)
        std::cout << std::setw(5) << l << "\t";
    std::cout << std::endl;
    for(Label l = 0; l <= maxLabel; ++l)
        std::cout << std::setw(5) << wCur[l] << "\t";
    std::cout << std::endl << std::endl;

    // Weight for second pixel
    std::cout << "w2" << std::endl;
    for(Label l = 0; l <= maxLabel; ++l)
        std::cout << std::setw(5) << l << "\t";
    std::cout << std::endl;
    for(Label l = 0; l <= maxLabel; ++l)
        std::cout << std::setw(5) << wCur[3 + l] << "\t";
    std::cout << std::endl << std::endl;

    // Pairwise weights
    for(Label l1 = 0; l1 <= maxLabel; ++l1)
    {
        for(Label l2 = 0; l2 <= maxLabel; ++l2)
        {
            std::cout << "w_{" << l1 << "," << l2 << "}" << std::endl;
            std::cout << std::setw(5) << wCur[6 + l1 + l2 * 3 + 0 * 3 * 2] << "\t";
            std::cout << std::setw(5) << wCur[6 + l1 + l2 * 3 + 1 * 3 * 2];
            std::cout << std::endl << std::endl;
        }
    }

    return 0;
}