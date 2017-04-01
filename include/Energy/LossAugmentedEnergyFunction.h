//
// Created by jan on 29.08.16.
//

#ifndef HSEG_LOSSAUGMENTEDENERGYFUNCTION_H
#define HSEG_LOSSAUGMENTEDENERGYFUNCTION_H

#include "EnergyFunction.h"

/**
 * Normal energy function, but with an added unary term for the hamming loss
 * @warning This class inherits from EnergyFunction, however, it is not meant to be used in a polymorphic context.
 */
class LossAugmentedEnergyFunction : private EnergyFunction
{
public:
    /**
     * Constructor
     * @param weights Weights to use. The pointer must stay valid as long as this object persists.
     * @param groundTruth Ground truth image. The pointer must stay valid as long as this object persists.
     * @param numClusters Amount of clusters
     */
    LossAugmentedEnergyFunction(Weights const* weights, LabelImage const* groundTruth, ClusterId numClusters, bool usePairwise = true);

    Cost giveEnergy(FeatureImage const& features, LabelImage const& labeling, LabelImage const& clustering, std::vector<Cluster> const& clusters) const;

    inline Cost unaryCost(SiteId i, Feature const& f, Label l) const
    {
        Cost loss = 0;
        if (m_pGroundTruth->atSite(i) != l && m_pGroundTruth->atSite(i) < numClasses())
            loss = m_lossFactor;

        return EnergyFunction::unaryCost(i, f, l) - loss;
    }

    inline Cost higherOrderSpecialUnaryCost(SiteId i, Label l_k) const
    {
        Cost loss = 0;
        if(m_pGroundTruth->atSite(i) != l_k)
            loss = m_lossFactor;
        return -loss;
    }

    /**
     * @return Loss of a single misclassified pixel on this image
     */
    Cost lossFactor();

    /**
     * Computes the loss of a labeling
     * @param labeling Labeling to compute loss for
     * @param groundTruth Ground truth image
     * @param lossFactor Loss factor as computed by computeLossFactor()
     * @param numClasses Amount of classes
     * @return The loss
     */
    static Cost
    computeLoss(LabelImage const& labeling, LabelImage const& clustering,
                LabelImage const& groundTruth,
                std::vector<Cluster> const& clusters, Cost lossFactor,
                Label numClasses);

    /**
     * Computes the loss factor on an image
     * @param groundTruth Ground truth image
     * @param numClasses Amount of classes
     * @return The loss factor on this image
     */
    static Cost computeLossFactor(LabelImage const& groundTruth, Label numClasses);

    /*
     * Provide some functionality from EnergyFunction that also works for loss augmented energies.
     */
    using EnergyFunction::numClasses;
    using EnergyFunction::numClusters;
    using EnergyFunction::pairwiseCost;
    using EnergyFunction::higherOrderCost;
    using EnergyFunction::featureCost;
    using EnergyFunction::weights;
    using EnergyFunction::usePairwise;

private:
    LabelImage const* m_pGroundTruth;
    Cost m_lossFactor;
};


#endif //HSEG_LOSSAUGMENTEDENERGYFUNCTION_H
