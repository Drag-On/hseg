//
// Created by jan on 20.08.16.
//

#include <helper/coordinate_helper.h>
#include "Energy/EnergyFunction.h"

EnergyFunction::EnergyFunction(UnaryFile const& unaries)
        : m_unaryScores(unaries)
{
}

float EnergyFunction::giveUnaryEnergy(LabelImage const& labeling) const
{
    float unaryEnergy = 0;
    for (size_t i = 0; i < labeling.pixels(); ++i)
    {
        auto coords = helper::coord::siteTo2DCoordinate(i, m_unaryScores.width());
        unaryEnergy += m_unaryWeight * (-m_unaryScores.at(coords.first, coords.second, labeling.atSite(i)));
    }
    return unaryEnergy;
}

float EnergyFunction::featureDistance(Feature const& feature, Feature const& feature2) const
{
    return feature.sqDistanceTo(feature2);
}

float EnergyFunction::classDistance(Label l1, Label l2) const
{
    return simplePotts(l1, l2);
}

float EnergyFunction::pixelToClusterDistance(Feature const& fPx, Label lPx, Feature const& fCl, Label lCl) const
{
    return featureDistance(fPx, fCl) + m_spGamma * classDistance(lPx, lCl);
}

float EnergyFunction::unaryCost(size_t i, Label l) const
{
    auto coords = helper::coord::siteTo2DCoordinate(i, m_unaryScores.width());
    return m_unaryWeight * (-m_unaryScores.at(coords.first, coords.second, l));
}

Label EnergyFunction::numClasses() const
{
    return m_unaryScores.classes();
}

float EnergyFunction::classWeight() const
{
    return m_spGamma;
}
