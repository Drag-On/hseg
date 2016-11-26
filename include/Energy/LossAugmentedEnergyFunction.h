//
// Created by jan on 29.08.16.
//

#ifndef HSEG_LOSSAUGMENTEDENERGYFUNCTION_H
#define HSEG_LOSSAUGMENTEDENERGYFUNCTION_H

#include "EnergyFunction.h"

/**
 * Normal energy function, but with an added unary term for the loss
 */
class LossAugmentedEnergyFunction : public EnergyFunction
{
public:
    /**
     * Constructor
     * @param unaries Unary scores. The reference must stay valid as long as this object persists.
     * @param weights Weights to use. The reference must stay valid as long as this object persists.
     * @param pairwiseSigmaSq Sigma-Square inside of the exponential
     * @param featureWeights Feature weight matrix
     * @param groundTruth Ground truth image. The reference must stay valid as long as this object persists.
     */
    LossAugmentedEnergyFunction(UnaryFile const& unaries, WeightsVec const& weights, float pairwiseSigmaSq,
                                Matrix5f const& featureWeights, LabelImage const& groundTruth);

    inline float unaryCost(size_t i, Label l) const
    {
        float loss = 0;
        if (m_groundTruth.atSite(i) != l && m_groundTruth.atSite(i) < m_unaryScores.classes())
            loss = m_lossFactor;

        return EnergyFunction::unaryCost(i, l) - loss;
    }

    inline float pixelToClusterDistance(Feature const& fPx, Label lPx, std::vector<Cluster> const& cl, size_t clusterId) const
    {
        float spLoss = 0;
        if (m_groundTruth.at(fPx.x(), fPx.y()) != cl[clusterId].label &&
            m_groundTruth.at(fPx.x(), fPx.y()) < m_unaryScores.classes())
            spLoss = s_spLossWeight * m_lossFactor;
        return EnergyFunction::pixelToClusterDistance(fPx, lPx, cl, clusterId) - spLoss;
    }

    /**
     * Additional cost to choose a label as the cluster representative. Only used for loss-augmented prediction.
     * @param idx Index of the pixel
     * @param l Label to assign
     * @return The additional cost of assigning \p l to a cluster
     */
    inline float additionalClassLabelCost(size_t idx, Label l) const
    {
        Label gtLabel = m_groundTruth.atSite(idx);
        if (gtLabel == l || gtLabel >= m_unaryScores.classes() || l >= m_unaryScores.classes())
            return 0;
        else
            return s_spLossWeight * m_lossFactor;
    }

    /**
     * Additional label used during loss-augmented clustering
     * @param idx Pixel index
     * @return Return the additional label at pixel \p idx
     */
    inline Label additionalLabel(size_t idx) const
    {
        return m_groundTruth.atSite(idx);
    }

    /**
     * @return Loss of a single misclassified pixel on this image
     */
    float lossFactor();

    /**
     * Computes the loss of a labeling
     * @param labeling Labeling to compute loss for
     * @param superpixels Superpixel labeling
     * @param groundTruth Ground truth image
     * @param lossFactor Loss factor as computed by computeLossFactor()
     * @param clusters Clusters of the superpixels
     * @param numClasses Amount of classes
     * @return The loss
     */
    static float
    computeLoss(LabelImage const& labeling, LabelImage const& superpixels, LabelImage const& groundTruth,
                float lossFactor, std::vector<Cluster> const& clusters, size_t numClasses);

    /**
     * Computes the loss factor on an image
     * @param groundTruth Ground truth image
     * @param numClasses Amount of classes
     * @return The loss factor on this image
     */
    static float computeLossFactor(LabelImage const& groundTruth, size_t numClasses);

private:
    LabelImage const& m_groundTruth;
    float m_lossFactor;
    static float constexpr s_spLossWeight = 0.5f;
};


#endif //HSEG_LOSSAUGMENTEDENERGYFUNCTION_H
