//
// Created by jan on 29.08.16.
//

#include "Energy/LossAugmentedEnergyFunction.h"

LossAugmentedEnergyFunction::LossAugmentedEnergyFunction(UnaryFile const& unaries, WeightsVec const& weights,
                                                         float pairwiseSigmaSq, Matrix5f const& featureWeights,
                                                         LabelImage const& groundTruth, LabelImage const& spGroundTruth)
        : EnergyFunction(unaries, weights, pairwiseSigmaSq, featureWeights),
          m_groundTruth(groundTruth),
          m_spGroundTruth(spGroundTruth)
{
    m_lossFactor = 0;
    for(size_t i = 0; i < m_groundTruth.pixels(); ++i)
        if(m_groundTruth.atSite(i) < m_unaryScores.classes())
            m_lossFactor++;
    m_lossFactor = 1e8f / m_lossFactor;
}

float LossAugmentedEnergyFunction::unaryCost(size_t i, Label l) const
{
    float loss = 0;
    if (m_groundTruth.atSite(i) != l && m_groundTruth.atSite(i) < m_unaryScores.classes())
        loss = m_lossFactor;

    return EnergyFunction::unaryCost(i, l) - loss;
}

float LossAugmentedEnergyFunction::pixelToClusterDistance(Feature const& fPx, Label lPx, std::vector<Cluster> const& cl, size_t clusterId) const
{
    float dist = EnergyFunction::pixelToClusterDistance(fPx, lPx, cl, clusterId);
    if(clusterId != m_spGroundTruth.at(fPx.x(), fPx.y()))
        dist -= m_lossFactor;
    return dist;
}

