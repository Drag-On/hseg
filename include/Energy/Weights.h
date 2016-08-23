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

    Weight unary(Label l) const;

    Weight pairwise(Label l1, Label l2) const;

    Weight pairwiseSigmaSq() const;

    FeatureWeights const& featureWeights() const;

    Weight classWeight() const;
};


#endif //HSEG_WEIGHTS_H
