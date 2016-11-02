//
// Created by jan on 20.08.16.
//

#include <helper/coordinate_helper.h>
#include "Energy/EnergyFunction.h"

EnergyFunction::EnergyFunction(UnaryFile const& unaries, WeightsVec const& weights, float pairwiseSigmaSq, Matrix5f const& featureWeights)
        : m_unaryScores(unaries),
          m_weights(weights),
          m_pairWiseSigmaSq(pairwiseSigmaSq),
          m_featureWeights(featureWeights)
{
}

void EnergyFunction::computeUnaryEnergyByWeight(LabelImage const& labeling, WeightsVec& energyW) const
{
    for (size_t i = 0; i < labeling.pixels(); ++i)
    {
        Label l = labeling.atSite(i);
        if (l < m_unaryScores.classes())
            energyW.m_unaryWeights[l] += -m_unaryScores.atSite(i, l);
    }
}

float EnergyFunction::unaryCost(size_t i, Label l) const
{
    if(l >= m_unaryScores.classes())
        return 0;

    auto coords = helper::coord::siteTo2DCoordinate(i, m_unaryScores.width());
    return m_weights.unary(l) * (-m_unaryScores.at(coords.x(), coords.y(), l));
}

float EnergyFunction::pixelToClusterDistance(Feature const& fPx, Label lPx, std::vector<Cluster> const& cl, size_t clusterId) const
{
    return featureDistance(fPx, cl[clusterId].mean) + classDistance(lPx, cl[clusterId].label);
}
