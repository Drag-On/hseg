//
// Created by jan on 29.08.16.
//

#ifndef HSEG_LOSSAUGMENTEDENERGYFUNCTION_H
#define HSEG_LOSSAUGMENTEDENERGYFUNCTION_H

#include "EnergyFunction.h"

/**
 * Normal energy function, but with an added unary term for the hamming loss
 * @warning This class inherits from EnergyFunction, however, it is not meant to be used in polymorphic ways.
 */
class LossAugmentedEnergyFunction : public EnergyFunction
{
public:
    /**
     * Constructor
     * @param weights Weights to use. The pointer must stay valid as long as this object persists.
     * @param groundTruth Ground truth image. The pointer must stay valid as long as this object persists.
     */
    LossAugmentedEnergyFunction(Weights const* weights, LabelImage const* groundTruth);

    inline Cost unaryCost(SiteId i, Feature const& f, Label l) const
    {
        Cost loss = 0;
        if (m_pGroundTruth->atSite(i) != l && m_pGroundTruth->atSite(i) < numClasses())
            loss = m_lossFactor;

        return EnergyFunction::unaryCost(i, f, l) - loss;
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
    computeLoss(LabelImage const& labeling, LabelImage const& groundTruth, Cost lossFactor, Label numClasses);

    /**
     * Computes the loss factor on an image
     * @param groundTruth Ground truth image
     * @param numClasses Amount of classes
     * @return The loss factor on this image
     */
    static Cost computeLossFactor(LabelImage const& groundTruth, Label numClasses);

private:
    LabelImage const* m_pGroundTruth;
    Cost m_lossFactor;

    /*
     * Make some functions private that are not really meant to be used via an object of this type.
     */

    using EnergyFunction::giveEnergy;
    using EnergyFunction::giveEnergyByWeight;
    using EnergyFunction::computeUnaryEnergyByWeight;
    using EnergyFunction::computePairwiseEnergyByWeight;
};


#endif //HSEG_LOSSAUGMENTEDENERGYFUNCTION_H
