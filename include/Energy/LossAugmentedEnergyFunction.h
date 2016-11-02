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

    inline float unaryCost(size_t i, Label l) const
    {
        float loss = 0;
        if (m_groundTruth.atSite(i) != l && m_groundTruth.atSite(i) < m_unaryScores.classes())
            loss = m_lossFactor;

        return EnergyFunction::unaryCost(i, l) - loss;
    }

    inline float pixelToClusterDistance(Feature const& fPx, Label lPx, std::vector<Cluster> const& cl, size_t clusterId) const
    {
        float dist = EnergyFunction::pixelToClusterDistance(fPx, lPx, cl, clusterId);
        if(clusterId != m_spGroundTruth.at(fPx.x(), fPx.y()))
            dist -= m_lossFactor;
        return dist;
    }

private:
    LabelImage const& m_groundTruth;
    LabelImage const& m_spGroundTruth;
    float m_lossFactor;
};


#endif //HSEG_LOSSAUGMENTEDENERGYFUNCTION_H
