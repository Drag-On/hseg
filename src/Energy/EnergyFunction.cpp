//
// Created by jan on 20.08.16.
//

#include <helper/coordinate_helper.h>
#include "Energy/EnergyFunction.h"

EnergyFunction::EnergyFunction(UnaryFile const& unaries, Weights const& weights)
        : m_unaryScores(unaries),
          m_weights(weights)
{
}

float EnergyFunction::giveUnaryEnergy(LabelImage const& labeling) const
{
    float unaryEnergy = 0;
    for (size_t i = 0; i < labeling.pixels(); ++i)
        unaryEnergy += unaryCost(i, labeling.atSite(i));
    return unaryEnergy;
}

float EnergyFunction::featureDistance(Feature const& feature, Feature const& feature2) const
{
    auto const xDiff = feature.x() - feature2.x();
    auto const yDiff = feature.y() - feature2.y();
    auto const rDiff = feature.r() - feature2.r();
    auto const gDiff = feature.g() - feature2.g();
    auto const bDiff = feature.b() - feature2.b();
    auto const& w = m_weights.featureWeights();
    auto const colorDist = w.a * (rDiff * rDiff + gDiff * gDiff + bDiff * bDiff);
    auto const spatialDist = w.b * (xDiff * xDiff) + w.c * (yDiff * yDiff) + 2 * w.d * (xDiff * yDiff);
    return colorDist + spatialDist;
}
