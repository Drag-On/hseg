//
// Created by jan on 26.08.16.
//

#include "Accuracy/ConfusionMatrix.h"

ConfusionMatrix::ConfusionMatrix(size_t numClasses) noexcept
        : m_numClasses(numClasses),
          m_mat(std::vector<size_t>(numClasses * numClasses, 0))
{
}

ConfusionMatrix::ConfusionMatrix(size_t numClasses, LabelImage const& labeling, LabelImage const& groundTruth)
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
    size_t max = *(std::max_element(m_mat.begin(), m_mat.end()));
    for(Label i = 0; i < m_numClasses; ++i)
    {
        for(Label j = 0; j < m_numClasses; ++j)
            img.at<unsigned char>(i, j) = at(i, j) * 255 / max;
    }
    return img;
}

std::ostream& operator<<(std::ostream& stream, ConfusionMatrix const& cf)
{
    float mean = 0;
    auto acc = cf.accuracies(&mean);

    stream << "Mean = " << mean << "; Per class = ";
    stream << "{ ";
    for (size_t i = 0; i < acc.size() - 1; ++i)
        stream << acc[i] << ", ";
    if (!acc.empty())
        stream << acc.back();
    stream << " }";
    return stream;
}
