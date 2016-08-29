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
    LossAugmentedEnergyFunction(UnaryFile const& unaries, Weights const& weights, float pairwiseSigmaSq, LabelImage const& groundTruth);

    virtual float unaryCost(size_t i, Label l) const override;

private:
    LabelImage const& m_groundTruth;
    float m_lossFactor;
};


#endif //HSEG_LOSSAUGMENTEDENERGYFUNCTION_H
