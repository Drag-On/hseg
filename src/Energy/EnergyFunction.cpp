//
// Created by jan on 20.08.16.
//

#include <helper/coordinate_helper.h>
#include "Energy/EnergyFunction.h"

EnergyFunction::EnergyFunction(UnaryFile const& unaries, WeightsVec const& weights, float pairwiseSigmaSq)
        : m_unaryScores(unaries),
          m_weights(weights),
          m_pairWiseSigmaSq(pairwiseSigmaSq)
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

float EnergyFunction::pairwisePixelWeight(CieLabImage const& img, size_t i, size_t j) const
{
    float rDiff = img.atSite(i, 0) - img.atSite(j, 0);
    float gDiff = img.atSite(i, 1) - img.atSite(j, 1);
    float bDiff = img.atSite(i, 2) - img.atSite(j, 2);
    float colorDiffNormSq = rDiff * rDiff + gDiff * gDiff + bDiff * bDiff;
    float weight = std::exp(-m_pairWiseSigmaSq * colorDiffNormSq);
    return weight;
}

float EnergyFunction::pairwiseClassWeight(Label l1, Label l2) const
{
    if (l1 >= m_unaryScores.classes() || l2 >= m_unaryScores.classes())
        return 0;
    else
        return m_weights.pairwise(l1, l2);
}

float EnergyFunction::featureDistance(Feature const& feature, Feature const& feature2) const
{
    auto const xDiff = feature.x() - feature2.x();
    auto const yDiff = feature.y() - feature2.y();
    auto const rDiff = feature.r() - feature2.r();
    auto const gDiff = feature.g() - feature2.g();
    auto const bDiff = feature.b() - feature2.b();
    auto const& w = m_weights.feature();
    auto const colorDist = w.a() * (rDiff * rDiff + gDiff * gDiff + bDiff * bDiff);
    auto const spatialDist = w.b() * (xDiff * xDiff) + w.c() * (yDiff * yDiff) + 2 * w.d() * (xDiff * yDiff);
    return colorDist + spatialDist;
}

void EnergyFunction::computeFeatureDistanceByWeight(Feature const& feature, Feature const& feature2,
                                                    WeightsVec& energyW) const
{
    auto const xDiff = feature.x() - feature2.x();
    auto const yDiff = feature.y() - feature2.y();
    auto const rDiff = feature.r() - feature2.r();
    auto const gDiff = feature.g() - feature2.g();
    auto const bDiff = feature.b() - feature2.b();
    Weight new_a = energyW.m_featureWeights.a() + rDiff * rDiff + gDiff * gDiff + bDiff * bDiff;
    Weight new_b = energyW.m_featureWeights.b() + xDiff * xDiff;
    Weight new_c = energyW.m_featureWeights.c() + yDiff * yDiff;
    Weight new_d = energyW.m_featureWeights.d() + 2 * (xDiff * yDiff);
    energyW.m_featureWeights.set(new_a, new_b, new_c, new_d);
}

float EnergyFunction::classDistance(Label l1, Label l2) const
{
    if (l1 == l2 || l1 >= m_unaryScores.classes() || l2 >= m_unaryScores.classes())
        return 0;
    else
        return m_weights.classWeight();
}

float EnergyFunction::pixelToClusterDistance(Feature const& fPx, Label lPx, Cluster const& cl) const
{
    return featureDistance(fPx, cl.mean) + classDistance(lPx, cl.label);
}

UnaryFile const& EnergyFunction::unaryFile() const
{
    return m_unaryScores;
}

WeightsVec const& EnergyFunction::weights() const
{
    return m_weights;
}
