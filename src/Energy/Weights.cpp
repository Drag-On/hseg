//
// Created by jan on 23.08.16.
//

#include "Energy/Weights.h"

Weights::Weights(Label numLabels, bool defaultInit)
        : m_numLabels(numLabels)
{
    if(defaultInit)
    {
        m_unaryWeights.resize(numLabels, 5.f);
        m_pairwiseWeights.resize((numLabels * numLabels) / 2, 300.f);
        m_featureWeights = { 0.2f, 0.8f, 0.8f, 0.f };
        m_classWeight = 30.f;
    }
    else
    {
        m_unaryWeights.resize(numLabels, 0.f);
        m_pairwiseWeights.resize((numLabels * numLabels) / 2, 0.f);
        m_featureWeights = { 0.f, 0.f, 0.f, 0.f };
        m_classWeight = 0.f;
    }
}

Weights::Weights(Label numLabels, float unaryWeight, float pairwiseWeight, float featA, float featB, float featC,
                 float featD, float labelWeight)
        : m_numLabels(numLabels)
{
    m_unaryWeights.resize(numLabels, unaryWeight);
    m_pairwiseWeights.resize((numLabels * numLabels) / 2, pairwiseWeight);
    m_featureWeights = {featA, featB, featC, featD};
    m_classWeight = labelWeight;
}

Weight Weights::pairwise(Label l1, Label l2) const
{
    // Diagonal is always zero
    if(l1 == l2)
        return 0;

    // The weight l1->l2 is the same as l2->l1
    if(l2 < l1)
        std::swap(l1, l2);

    // Pairwise indices are stored as upper triangular matrix
    size_t const index = l1 + l2 * (l2 - 1) / 2;
    assert(index < m_pairwiseWeights.size());
    return m_pairwiseWeights[index];
}

