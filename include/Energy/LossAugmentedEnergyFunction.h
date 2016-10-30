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
     * @param spGroundTruth Ground truth image of the superpixel segmentation.
     */
    LossAugmentedEnergyFunction(UnaryFile const& unaries, WeightsVec const& weights, float pairwiseSigmaSq,
                                Matrix5f const& featureWeights, LabelImage const& groundTruth, LabelImage const& spGroundTruth);

    virtual float unaryCost(size_t i, Label l) const override;

    float pixelToClusterDistance(Feature const& fPx, Label lPx, Cluster const& cl, size_t clusterId) const override;

private:
    LabelImage const& m_groundTruth;
    LabelImage const& m_spGroundTruth;
    float m_lossFactor;
};


#endif //HSEG_LOSSAUGMENTEDENERGYFUNCTION_H
