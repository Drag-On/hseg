//
// Created by jan on 23.08.16.
//

#ifndef HSEG_WEIGHTS_H
#define HSEG_WEIGHTS_H


#include <vector>
#include <Image/Image.h>

using Weight = float;

/**
 * Contains all (trainable) weights needed by the energy function
 */
class Weights
{
private:
    Label m_numLabels;
    std::vector<Weight> m_unaryWeights;
    std::vector<Weight> m_pairwiseWeights;
    struct FeatureWeights
    {
        Weight a, b, c, d;
    } m_featureWeights;
    Weight m_classWeight;

public:
    /**
     * Creates default weights
     * @param numLabels Amount of class labels
     * @param defaultInit Determines if the weights will be initialized with default values or with zeros
     */
    explicit Weights(Label numLabels, bool defaultInit = true);

    /**
     * Initialize some basic weights
     * @param numLabels Amount of class labels
     * @param unaryWeight Weight of the unary term
     * @param pairwiseWeight Weight of the pairwise term
     * @param featA Color weight
     * @param featB X distance weight
     * @param featC Y distance weight
     * @param featD XY distance weight
     * @param labelWeight Label weight
     */
    Weights(Label numLabels, float unaryWeight, float pairwiseWeight, float featA, float featB, float featC, float featD, float labelWeight);

    /**
     * Weight of the unary term
     * @param l Class label
     * @return The approriate weight
     */
    inline Weight unary(Label l) const
    {
        assert(l < m_unaryWeights.size());
        return m_unaryWeights[l];
    }

    /**
     * Weight of the pairwise term
     * @param l1 First label
     * @param l2 Second label
     * @return The approriate weight
     */
    Weight pairwise(Label l1, Label l2) const;

    /**
     * The feature weights.
     * @details a is the factor for all 3 color channels, b and c are factors for x and y respectively, and d is the
     *          off-diagonal for the spatial feature
     * @return The feature weights
     */
    inline FeatureWeights const& featureWeights() const
    {
        return m_featureWeights;
    }

    /**
     * @return The class weight
     */
    inline Weight classWeight() const
    {
        return m_classWeight;
    }
};


#endif //HSEG_WEIGHTS_H
