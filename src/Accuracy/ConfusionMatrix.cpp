//
// Created by jan on 26.08.16.
//

#include "Accuracy/ConfusionMatrix.h"

ConfusionMatrix::ConfusionMatrix(Label numClasses) noexcept
        : m_numClasses(numClasses),
          m_mat(std::vector<size_t>(numClasses * numClasses, 0))
{
}

ConfusionMatrix::ConfusionMatrix(Label numClasses, LabelImage const& labeling, LabelImage const& groundTruth)
        : m_numClasses(numClasses),
          m_mat(std::vector<size_t>(numClasses * numClasses, 0))
{
    assert(labeling.width() == groundTruth.width() && labeling.height() == groundTruth.height());

    for (size_t i = 0; i < labeling.pixels(); ++i)
    {
        Label inferredLabel = labeling.atSite(i);
        Label trueLabel = groundTruth.atSite(i);

        if (inferredLabel < m_numClasses && trueLabel < m_numClasses)
            at(trueLabel, inferredLabel)++;
    }
}

size_t ConfusionMatrix::at(Label trueLabel, Label inferredLabel) const
{
    return m_mat[trueLabel + inferredLabel * m_numClasses];
}

size_t& ConfusionMatrix::at(Label trueLabel, Label inferredLabel)
{
    return m_mat[trueLabel + inferredLabel * m_numClasses];
}

std::vector<float> ConfusionMatrix::accuracies(float* mean) const
{
    std::vector<float> accuracies(m_numClasses, 0.f);
    std::vector<size_t> trueSums(m_numClasses, 0);
    std::vector<size_t> predictedSums(m_numClasses, 0);
    for (Label predicted = 0; predicted < m_numClasses; ++predicted)
    {
        for (Label truth = 0; truth < m_numClasses; ++truth)
        {
            trueSums[predicted] += at(truth, predicted);
            predictedSums[truth] += at(truth, predicted);
        }
    }
    for (Label l = 0; l < m_numClasses; ++l)
    {
        float truePositives = at(l, l);
        float falsePositives = predictedSums[l] - truePositives;
        float falseNegatives = trueSums[l] - truePositives;
        accuracies[l] = truePositives / (truePositives + falsePositives + falseNegatives);
    }

    if (mean != nullptr)
    {
        *mean = 0.f;
        size_t nonNullEntries = 0;
        for (auto const& a : accuracies)
        {
            if (!std::isnan(a))
            {
                nonNullEntries++;
                *mean += a;
            }
        }
        *mean = *mean / nonNullEntries;
    }

    return accuracies;
}

void ConfusionMatrix::join(LabelImage const& labeling, LabelImage const& groundTruth)
{
    for (size_t i = 0; i < labeling.pixels(); ++i)
    {
        Label inferredLabel = labeling.atSite(i);
        Label trueLabel = groundTruth.atSite(i);

        if (inferredLabel < m_numClasses && trueLabel < m_numClasses)
            at(trueLabel, inferredLabel)++;
    }
}

ConfusionMatrix::operator cv::Mat() const
{
    cv::Mat img(m_numClasses, m_numClasses, CV_8UC1);
    // Compute row and column sums
    /*std::vector<size_t> trueSum(m_numClasses, 0);
    for(Label i = 0; i < m_numClasses; ++i) // true label
    {
        trueSum[i] = at(i, 0);
        for(Label j = 1; j < m_numClasses; ++j) // predicted label
            trueSum[i] += at(i, j);
    }*/
    std::vector<size_t> predSum(m_numClasses, 0);
    for(Label j = 0; j < m_numClasses; ++j) // predicted label
    {
        predSum[j] = at(0, j);
        for(Label i = 1; i < m_numClasses; ++i) // true label
            predSum[j] += at(i, j);
    }
    // Normalize matrix
    for(Label i = 0; i < m_numClasses; ++i) // true label
    {
        for(Label j = 0; j < m_numClasses; ++j) // predicted label
        {
            size_t sum = predSum[i];
            if(sum > 0)
                img.at<uchar>(i, j) = at(i, j) * 255 / sum;
            else
                img.at<uchar>(i, j) = 0;
        }
    }
    return img;
}

std::ostream& operator<<(std::ostream& stream, ConfusionMatrix const& cf)
{
    // IoU measure
    float mean = 0;
    auto acc = cf.accuracies(&mean);

    stream << "Mean = " << mean << "; Per class = ";
    stream << "{ ";
    for (size_t i = 0; i < acc.size() - 1; ++i)
        stream << acc[i] << ", ";
    if (!acc.empty())
        stream << acc.back();
    stream << " }; ";

    // Overall percentage
    float diag = 0;
    for(size_t i = 0; i < acc.size(); ++i)
        diag += cf.at(i, i);
    float all = std::accumulate(cf.m_mat.begin(), cf.m_mat.end(), 0.f);
    float overallAcc = diag / all;
    stream << "Overall percentage: " << overallAcc;

    return stream;
}
