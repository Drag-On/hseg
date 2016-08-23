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
    auto xDiff = feature.x() - feature2.x();
    auto yDiff = feature.y() - feature2.y();
    auto rDiff = feature.r() - feature2.r();
    auto gDiff = feature.g() - feature2.g();
    auto bDiff = feature.b() - feature2.b();
    auto const& w = m_weights.featureWeights();
    auto colorDist = w.a * (rDiff * rDiff + gDiff * gDiff + bDiff * bDiff);
    auto spatialDist = w.b * (xDiff * xDiff) + w.c * (yDiff * yDiff) + 2 * w.d * (xDiff * yDiff);
    return colorDist + spatialDist;
}

float EnergyFunction::classDistance(Label l1, Label l2) const
{
    if (l1 == l2)
        return 0;
    else
        return m_weights.classWeight();
}

float EnergyFunction::pixelToClusterDistance(Feature const& fPx, Label lPx, Feature const& fCl, Label lCl) const
{
    return featureDistance(fPx, fCl) + classDistance(lPx, lCl);
}

float EnergyFunction::unaryCost(size_t i, Label l) const
{
    auto coords = helper::coord::siteTo2DCoordinate(i, m_unaryScores.width());
    return m_weights.unary(l) * (-m_unaryScores.at(coords.x(), coords.y(), l));
}

float EnergyFunction::pairwiseClassWeight(Label l1, Label l2) const
{
    return m_weights.pairwise(l1, l2);
}

Label EnergyFunction::numClasses() const
{
    return static_cast<Label>(m_unaryScores.classes());
}
