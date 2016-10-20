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

float EnergyFunction::featureDistance(Feature const& feature, Feature const& feature2) const
{
    Vector5f f;
    f(0) = feature.r() - feature2.r();
    f(1) = feature.g() - feature2.g();
    f(2) = feature.b() - feature2.b();
    f(3) = feature.x() - feature2.x();
    f(4) = feature.y() - feature2.y();
    float dist = f.transpose() * m_featureWeights * f;
    return dist;
}

UnaryFile const& EnergyFunction::unaryFile() const
{
    return m_unaryScores;
}

WeightsVec const& EnergyFunction::weights() const
{
    return m_weights;
}
