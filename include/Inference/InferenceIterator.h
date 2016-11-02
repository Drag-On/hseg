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
template<typename EnergyFun>
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
    InferenceIterator(EnergyFun e, size_t numClusters, size_t numClasses, CieLabImage const& color);

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
    EnergyFun m_energy;
    size_t m_numClusters;
    size_t m_numClasses;
    CieLabImage const& m_color;

    float computeInitialEnergy(LabelImage const& labeling) const;
};

template<typename EnergyFun>
InferenceIterator<EnergyFun>::InferenceIterator(EnergyFun e, size_t numClusters, size_t numClasses,
                                                CieLabImage const& color)
        : m_energy(e),
          m_numClusters(numClusters),
          m_numClasses(numClasses),
          m_color(color)
{
}

template<typename EnergyFun>
InferenceResult InferenceIterator<EnergyFun>::run(size_t numIter)
{
    InferenceResult result;
    Clusterer<EnergyFun> clusterer(m_energy);
    GraphOptimizer<EnergyFun> optimizer(m_energy);

    result.labeling = m_energy.unaryFile().maxLabeling();
    for (size_t iter = 0; iter < numIter; ++iter)
    {
        // Update superpixels using the latest class labeling
        clusterer.run(m_numClusters, m_numClasses, m_color, result.labeling);
        result.superpixels = clusterer.clustership();

        // Update class labeling using the latest superpixels
        optimizer.run(m_color, result.superpixels, m_numClusters);
        result.labeling = optimizer.labeling();
    }

    return result;
}

template<typename EnergyFun>
InferenceResultDetails InferenceIterator<EnergyFun>::runDetailed(size_t numIter)
{
    Clusterer<EnergyFun> clusterer(m_energy);
    GraphOptimizer<EnergyFun> optimizer(m_energy);

    InferenceResultDetails result;

    LabelImage maxLabeling = m_energy.unaryFile().maxLabeling();
    float initialEnergy = computeInitialEnergy(maxLabeling);
    result.energy.push_back(initialEnergy);

    LabelImage spLabeling;
    LabelImage classLabeling = maxLabeling;

    for (size_t iter = 0; iter < numIter; ++iter)
    {
        // Update superpixels using the latest class labeling
        clusterer.run(m_numClusters, m_numClasses, m_color, classLabeling);
        spLabeling = clusterer.clustership();

        // Update class labeling using the latest superpixels
        optimizer.run(m_color, spLabeling, m_numClusters);
        classLabeling = optimizer.labeling();
        auto clusters = Clusterer<EnergyFun>::computeClusters(spLabeling, m_color, classLabeling, m_numClusters,
                                                              m_numClasses, m_energy);

        float energy = m_energy.giveEnergy(classLabeling, m_color, spLabeling, clusters);
        result.energy.push_back(energy);
        result.labelings.push_back(classLabeling);
        result.superpixels.push_back(spLabeling);
    }
    result.numIter = numIter;

    return result;
}

template<typename EnergyFun>
float InferenceIterator<EnergyFun>::computeInitialEnergy(LabelImage const& labeling) const
{
    LabelImage fakeSpLabeling(m_color.width(), m_color.height());
    Cluster c(&m_energy);
    for (size_t i = 0; i < m_color.pixels(); ++i)
    {
        c.accumFeature += Feature(m_color, i);
        c.labelFrequencies[labeling.atSite(i)]++;
    }
    c.size = m_color.pixels();
    c.updateMean();
    c.updateLabel();
    std::vector<Cluster> fakeClusters(1, c);

    float energy = m_energy.giveEnergy(labeling, m_color, fakeSpLabeling, fakeClusters);
    return energy;
}


#endif //HSEG_INFERENCEITERATOR_H
