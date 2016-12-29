//
// Created by jan on 20.08.16.
//

#include <helper/coordinate_helper.h>
#include "Energy/EnergyFunction.h"

EnergyFunction::EnergyFunction(UnaryFile const& unaries, WeightsVec const& weights, Cost pairwiseSigmaSq, Matrix5 const& featureWeights)
        : m_unaryScores(unaries),
          m_weights(weights),
          m_pairWiseSigmaSq(pairwiseSigmaSq),
          m_featureWeights(featureWeights)
{
}

void EnergyFunction::computeUnaryEnergyByWeight(LabelImage const& labeling, WeightsVec& energyW) const
{
    for (SiteId i = 0; i < labeling.pixels(); ++i)
    {
        Label l = labeling.atSite(i);
        if (l < m_unaryScores.classes())
            energyW.m_unaryWeights[l] += -m_unaryScores.atSite(i, l);
    }
}

void EnergyFunction::featureDistanceByWeight(Feature const& feature, Feature const& feature2, WeightsVec& weights) const
{
    Vector5 f = feature.vec() - feature2.vec();
    Matrix5 factors = f * f.transpose();
    for (uint16_t i = 0; i < 5; ++i)
    {
        for (uint16_t j = 0; j < 5; ++j)
            weights.featureWeight(i, j) += factors(i, j);
    }
}