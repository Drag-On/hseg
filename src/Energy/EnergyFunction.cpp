//
// Created by jan on 20.08.16.
//

#include <helper/coordinate_helper.h>
#include "Energy/EnergyFunction.h"

EnergyFunction::EnergyFunction(Weights const* weights, ClusterId numClusters)
        : m_pWeights(weights),
          m_numClusters(numClusters)
{
}

Cost EnergyFunction::giveEnergy(FeatureImage const& features, LabelImage const& labeling) const
{
    Weights energy = giveEnergyByWeight(features, labeling);
    return (*m_pWeights) * energy;
}

Weights EnergyFunction::giveEnergyByWeight(FeatureImage const& features, LabelImage const& labeling) const
{
    Weights w(numClasses(), features.dim()); // Zero-initialized weights

    computeUnaryEnergyByWeight(features, labeling, w);
    computePairwiseEnergyByWeight(features, labeling, w);

    return w;
}

void EnergyFunction::computeUnaryEnergyByWeight(FeatureImage const& features, LabelImage const& labeling, Weights& energyW) const
{
    for (SiteId i = 0; i < labeling.pixels(); ++i)
    {
        Label l = labeling.atSite(i);
        Feature const& f = features.atSite(i);
        Feature combinedFeat(f.size() + 1);
        combinedFeat << f, 1.f;
        energyW.m_unaryWeights[l] += combinedFeat;
    }
}

void EnergyFunction::computePairwiseEnergyByWeight(FeatureImage const& features, LabelImage const& labeling, Weights& energyW) const
{
    for (Coord x = 0; x < labeling.width(); ++x)
    {
        for (Coord y = 0; y < labeling.height(); ++y)
        {
            Label l = labeling.at(x, y);
            Feature const& f = features.at(x, y);
            if(x + 1 < labeling.width())
            {
                Label lR = labeling.at(x + 1, y);
                Feature const& fR = features.at(x + 1, y);

                Feature combinedFeat(f.size() + fR.size() + 1);
                combinedFeat << f, fR, 1.f;
                energyW.pairwise(l, lR) += combinedFeat;
            }

            if(y + 1 < labeling.height())
            {
                Label lD = labeling.at(x, y + 1);
                Feature const& fD = features.at(x, y + 1);

                Feature combinedFeat(f.size() + fD.size() + 1);
                combinedFeat << f, fD, 1.f;
                energyW.pairwise(l, lD) += combinedFeat;
            }
        }
    }
}