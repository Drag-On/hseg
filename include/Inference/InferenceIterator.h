//
// Created by jan on 28.08.16.
//

#ifndef HSEG_INFERENCEITERATOR_H
#define HSEG_INFERENCEITERATOR_H

#include <Energy/EnergyFunction.h>
#include <Inference/InferenceResult.h>
#include "InferenceResultDetails.h"

/**
 * Infers both class labels and superpixels on an image
 */
class InferenceIterator
{
public:
    /**
     * Constructor
     * @param e Energy function
     * @param numClusters Amount of clusters
     * @param numClasses Amount of classes
     * @param color Color image
     */
    InferenceIterator(EnergyFunction const& e, size_t numClusters, size_t numClasses, CieLabImage const& color);

    /**
     * Does the actual inference
     * @param numIter Amount of iterations to do
     * @return Resulting labeling and segmentation
     */
    InferenceResult run(size_t numIter = 4);

    /**
     * Does inference and saves detailed results
     * @param numIter Amount of iterations to do
     * @return Detailed results
     */
    InferenceResultDetails runDetailed(size_t numIter = 4);

private:
    EnergyFunction const& m_energy;
    size_t m_numClusters;
    size_t m_numClasses;
    CieLabImage const& m_color;

    float computeInitialEnergy(LabelImage const& labeling) const;
};


#endif //HSEG_INFERENCEITERATOR_H
