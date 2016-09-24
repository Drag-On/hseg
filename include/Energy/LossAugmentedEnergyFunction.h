//
// Created by jan on 29.08.16.
//

#ifndef HSEG_LOSSAUGMENTEDENERGYFUNCTION_H
#define HSEG_LOSSAUGMENTEDENERGYFUNCTION_H

#include "EnergyFunction.h"

/**
 * Normal energy function, but with an added unary term for the hamming lass
 */
class LossAugmentedEnergyFunction : public EnergyFunction
{
public:
    /**
     * Constructor
     * @param unaries Unary scores. The reference must stay valid as long as this object persists.
     * @param weights Weights to use. The reference must stay valid as long as this object persists.
     * @param pairwiseSigmaSq Sigma-Square inside of the exponential
     * @param groundTruth Ground truth image. The reference must stay valid as long as this object persists.
     */
    LossAugmentedEnergyFunction(UnaryFile const& unaries, WeightsVec const& weights, float pairwiseSigmaSq, LabelImage const& groundTruth);

    virtual float unaryCost(size_t i, Label l) const override;

    float pairwisePixelWeight(CieLabImage const& img, size_t i, size_t j) const override;

    float pairwiseClassWeight(Label l1, Label l2) const override;

    float classDistance(Label l1, Label l2) const override;

    float pixelToClusterDistance(Feature const& fPx, Label lPx, Cluster const& cl) const override;

private:
    LabelImage const& m_groundTruth;
    float m_lossFactor;
    float m_constant = 1000000;
};


#endif //HSEG_LOSSAUGMENTEDENERGYFUNCTION_H
