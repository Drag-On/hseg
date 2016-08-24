//
// Created by jan on 23.08.16.
//

#ifndef HSEG_WEIGHTS_H
#define HSEG_WEIGHTS_H


#include <vector>
#include <Image/Image.h>

using Weight = float;

/**
 * Contains all weights needed by the energy function
 */
class Weights
{
private:
    Label m_numLabels;
    std::vector<Weight> m_unaryWeights;
    std::vector<Weight> m_pairwiseWeights;
    Weight m_pairwiseSigmaSq;
    struct FeatureWeights
    {
        Weight a, b, c, d;
    } m_featureWeights;
    Weight m_classWeight;

public:
    /**
     * Creates default weights
     * @param numLabels Amount of class labels
     */
    explicit Weights(Label numLabels);

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
     * @return The sigma-square in the exponential of the pairwise term
     */
    inline Weight pairwiseSigmaSq() const
    {
        return m_pairwiseSigmaSq;
    }

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