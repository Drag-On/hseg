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
    LossAugmentedEnergyFunction(UnaryFile const& unaries, WeightsVec const& weights, Cost pairwiseSigmaSq,
                                Matrix5 const& featureWeights, LabelImage const& groundTruth);

    inline Cost unaryCost(size_t i, Label l) const
    {
        Cost loss = 0;
        if (m_groundTruth.atSite(i) != l && m_groundTruth.atSite(i) < m_unaryScores.classes())
            loss = m_lossFactor;

        return EnergyFunction::unaryCost(i, l) - loss;
    }

    inline Cost classData(SiteId s, Label l) const
    {
        if(m_groundTruth.atSite(s) != l && m_groundTruth.atSite(s) < m_unaryScores.classes())
            return -m_lossFactor;
        else
            return 0;
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
    computeLoss(LabelImage const& labeling, LabelImage const& superpixels,
                LabelImage const& groundTruth, Cost lossFactor, Label numClasses,
                std::vector<Cluster> const& clustering);

    /**
     * Computes the loss factor on an image
     * @param groundTruth Ground truth image
     * @param numClasses Amount of classes
     * @return The loss factor on this image
     */
    static Cost computeLossFactor(LabelImage const& groundTruth, Label numClasses);

private:
    LabelImage const& m_groundTruth;
    Cost m_lossFactor;
};


#endif //HSEG_LOSSAUGMENTEDENERGYFUNCTION_H
