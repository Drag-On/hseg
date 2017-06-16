//
// Created by jan on 20.08.16.
//

#include "helper/coordinate_helper.h"
#include "Timer.h"
#include "Energy/EnergyFunction.h"

EnergyFunction::EnergyFunction(Weights const* weights, ClusterId numClusters, bool usePairwise)
        : m_pWeights(weights),
          m_numClusters(numClusters),
          m_usePairwise(usePairwise)
{
}

Cost EnergyFunction::giveEnergy(FeatureImage const& pxFeat, FeatureImage const& clusterFeat, LabelImage const& labeling, LabelImage const& clustering, std::vector<Cluster> const& clusters, LabelImage const* gt) const
{
    Weights energy = giveEnergyByWeight(pxFeat, clusterFeat, labeling, clustering, clusters, gt);
    return (*m_pWeights) * energy;
}

Weights EnergyFunction::giveEnergyByWeight(FeatureImage const& pxFeat, FeatureImage const& clusterFeat, LabelImage const& labeling, LabelImage const& clustering, std::vector<Cluster> const& clusters, LabelImage const* gt) const
{
    PROFILE_THIS

    Weights w(numClasses(), pxFeat.dim(), clusterFeat.dim()); // Zero-initialized weights

    computeUnaryEnergyByWeight(pxFeat, labeling, w, gt);
    if(m_usePairwise)
        computePairwiseEnergyByWeight(pxFeat, labeling, w, gt);
    computeHigherOrderEnergyByWeight(clusterFeat, labeling, clustering, clusters, w, gt);

    return w;
}

void EnergyFunction::computeUnaryEnergyByWeight(FeatureImage const& features, LabelImage const& labeling, Weights& energyW, LabelImage const* gt) const
{
    for (SiteId i = 0; i < labeling.pixels(); ++i)
    {
        // Skip invalid pixels
        if(gt && gt->atSite(i) >= numClasses())
            continue;

        Label l = labeling.atSite(i);
        if(l < numClasses())
        {
            Feature const& f = features.atSite(i);
            Feature combinedFeat(f.size() + 1);
            combinedFeat << f, 1.f;
            energyW.m_unaryWeights[l] += combinedFeat;
        }
    }
}

void EnergyFunction::computePairwiseEnergyByWeight(FeatureImage const& features, LabelImage const& labeling, Weights& energyW, LabelImage const* gt) const
{
    for (Coord x = 0; x < labeling.width(); ++x)
    {
        for (Coord y = 0; y < labeling.height(); ++y)
        {
            // Skip invalid pixels
            if(gt && gt->at(x, y) >= numClasses())
                continue;

            Label l = labeling.at(x, y);
            if(l >= numClasses())
                continue;
            Feature const& f = features.at(x, y);
            if(x + 1 < labeling.width())
            {
                // Skip invalid pixels
                if(gt && gt->at(x + 1, y) >= numClasses())
                    continue;

                Label lR = labeling.at(x + 1, y);
                if(lR >= numClasses())
                    continue;
                Feature const& fR = features.at(x + 1, y);

                Feature combinedFeat(f.size() + fR.size() + 1);
                combinedFeat << f, fR, 1.f;
                energyW.pairwise(l, lR) += combinedFeat;
            }

            if(y + 1 < labeling.height())
            {
                // Skip invalid pixels
                if(gt && gt->at(x, y + 1) >= numClasses())
                    continue;

                Label lD = labeling.at(x, y + 1);
                if(lD >= numClasses())
                    continue;
                Feature const& fD = features.at(x, y + 1);

                Feature combinedFeat(f.size() + fD.size() + 1);
                combinedFeat << f, fD, 1.f;
                energyW.pairwise(l, lD) += combinedFeat;
            }
        }
    }
}

void EnergyFunction::computeHigherOrderEnergyByWeight(FeatureImage const& features, LabelImage const& labeling,
                                                      LabelImage const& clustering,
                                                      std::vector<Cluster> const& clusters, Weights& energyW,
                                                      LabelImage const* gt) const
{
    if(numClusters() == 0)
        return;

    for(SiteId i = 0; i < labeling.pixels(); ++i)
    {
        // Skip invalid pixels
        if(gt && gt->atSite(i) >= numClasses())
            continue;

        Feature const& f = features.atSite(i);
        Label const l = labeling.atSite(i);
        if(l >= numClasses())
            continue;
        ClusterId const k = clustering.atSite(i);

        Feature const& fClus = clusters[k].m_feature;
        Label lClus = clusters[k].m_label;

        // Feature similarity
        auto diff = f - fClus;
        energyW.feature(l, lClus) += diff.cwiseProduct(diff);

        // Label consistency
        Feature combinedFeat(f.size() + fClus.size() + 1);
        combinedFeat << f, fClus, 1.f;
        energyW.higherOrder(l, lClus) += combinedFeat;
    }
}

void EnergyFunction::computeFeatureGradient(FeatureImage& outGradients, LabelImage const& labeling,
                                            LabelImage const& clustering, std::vector<Cluster> const& clusters,
                                            FeatureImage const& features) const
{
    assert(outGradients.width() == labeling.width());
    assert(outGradients.height() == labeling.height());
    assert(outGradients.dim() == m_pWeights->unary(0).size() - 1);
    assert(labeling.width() == clustering.width());
    assert(labeling.height() == clustering.height());

    unsigned int const featSize = outGradients.dim();

    for (SiteId i = 0; i < labeling.pixels(); ++i)
    {
        Feature& grad = outGradients.atSite(i);
        auto coords = helper::coord::siteTo2DCoordinate(i, labeling.width());
        Label l = labeling.atSite(i);
        if(l >= numClasses())
        {
            grad = Feature::Zero(featSize);
            continue;
        }

        // unary
        grad = m_pWeights->unary(l);

        // pairwise
        if(m_usePairwise)
        {
            if(static_cast<int>(coords.x()) - 1 >= 0)
            {
                Label l2 = labeling.at(coords.x() - 1, coords.y());
                if(l2 < numClasses())
                    grad += m_pWeights->pairwise(l2, l).tail(featSize);
            }
            if(coords.x() + 1 < labeling.width())
            {
                Label l2 = labeling.at(coords.x() + 1, coords.y());
                if(l2 < numClasses())
                    grad += m_pWeights->pairwise(l, l2).head(featSize);
            }
            if(static_cast<int>(coords.y()) - 1 >= 0)
            {
                Label l2 = labeling.at(coords.x(), coords.y() - 1);
                if(l2 < numClasses())
                    grad += m_pWeights->pairwise(l2, l).tail(featSize);
            }
            if(coords.y() + 1 < labeling.height())
            {
                Label l2 = labeling.at(coords.x(), coords.y() + 1);
                if(l2 < numClasses())
                    grad += m_pWeights->pairwise(l, l2).head(featSize);
            }
        }

        // higher-order
        if(numClusters() > 0)
        {
            Label l_clus = clusters[clustering.atSite(i)].m_label;
            grad += 2.f * m_pWeights->feature(l, l_clus) * (features.atSite(i) - clusters[clustering.atSite(i)].m_feature);
            grad += m_pWeights->higherOrder(l, l_clus).segment(0, featSize);
        }
    }
}
