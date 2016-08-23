//
// Created by jan on 23.08.16.
//

#include "Energy/Weights.h"

Weights::Weights(Label numLabels)
        : m_numLabels(numLabels)
{
    m_unaryWeights.resize(numLabels, 5.f);
    m_pairwiseWeights.resize((numLabels * numLabels) / 2, 300.f);
    m_pairwiseSigmaSq = 0.05f;
    m_featureWeights = { 0.3f, 0.7f, 0.7f, 0.f };
    m_classWeight = 30.f;
}

Weight Weights::unary(Label l) const
{
    assert(l < m_unaryWeights.size());
    return m_unaryWeights[l];
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
    size_t index = l1 + l2 * (l2 - 1) / 2;
    assert(index < m_pairwiseWeights.size());
    return m_pairwiseWeights[index];
}

Weight Weights::pairwiseSigmaSq() const
{
    return m_pairwiseSigmaSq;
}

Weights::FeatureWeights const& Weights::featureWeights() const
{
    return m_featureWeights;
}

Weight Weights::classWeight() const
{
    return m_classWeight;
}
