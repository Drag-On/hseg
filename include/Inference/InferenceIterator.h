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
template<typename EnergyFun, template<typename> class Optimizer = GraphOptimizer>
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
     * @param numIter Amount of iterations to do. If 0, run until convergence.
     * @return Resulting labeling and segmentation
     */
    InferenceResult run(size_t numIter = 0);

    /**
     * Does inference and saves detailed results
     * @param numIter Amount of iterations to do. If 0, run until convergence.
     * @return Detailed results
     */
    InferenceResultDetails runDetailed(size_t numIter = 0);

private:
    EnergyFun m_energy;
    size_t m_numClusters;
    size_t m_numClasses;
    CieLabImage const& m_color;
    float const m_eps = 1e3f;

    float computeInitialEnergy(LabelImage const& labeling) const;
};

template<typename EnergyFun, template<typename> class Optimizer>
InferenceIterator<EnergyFun, Optimizer>::InferenceIterator(EnergyFun e, size_t numClusters, size_t numClasses,
                                                CieLabImage const& color)
        : m_energy(e),
          m_numClusters(numClusters),
          m_numClasses(numClasses),
          m_color(color)
{
}

template<typename EnergyFun, template<typename> class Optimizer>
InferenceResult InferenceIterator<EnergyFun, Optimizer>::run(size_t numIter)
{
    InferenceResult result;
    Clusterer<EnergyFun> clusterer(m_energy);
    Optimizer<EnergyFun> optimizer(m_energy);

    float energy = std::numeric_limits<float>::max();
    float lastEnergy = energy;
    //result.labeling = m_energy.unaryFile().maxLabeling();
    result.labeling = LabelImage(m_color.width(), m_color.height()); // The labeling will be empty (all zeros)
    for (size_t iter = 0; (numIter > 0) ? (iter < numIter) : (lastEnergy - energy >= m_eps || iter == 0); ++iter)
    {
        lastEnergy = energy;

        // Update superpixels using the latest class labeling
        clusterer.run(m_numClusters, m_numClasses, m_color, result.labeling);
        result.superpixels = clusterer.clustership();

        // Update class labeling using the latest superpixels
        optimizer.run(m_color, result.superpixels, m_numClusters);
        result.labeling = optimizer.labeling();

        if(numIter == 0)
        {
            auto clusters = Clusterer<EnergyFun>::computeClusters(result.superpixels, m_color, result.labeling,
                                                                  m_numClusters, m_numClasses, m_energy);
            energy = m_energy.giveEnergy(result.labeling, m_color, result.superpixels, clusters);
        }
    }

    return result;
}

template<typename EnergyFun, template<typename> class Optimizer>
InferenceResultDetails InferenceIterator<EnergyFun, Optimizer>::runDetailed(size_t numIter)
{
    Clusterer<EnergyFun> clusterer(m_energy);
    GraphOptimizer<EnergyFun> optimizer(m_energy);

    InferenceResultDetails result;

    //LabelImage maxLabeling = m_energy.unaryFile().maxLabeling();
    LabelImage maxLabeling = LabelImage(m_color.width(), m_color.height()); // The labeling will be empty (all zeros)
    float initialEnergy = computeInitialEnergy(maxLabeling);
    float lastEnergy = initialEnergy;
    float energy = initialEnergy;
    result.energy.push_back(initialEnergy);

    LabelImage spLabeling;
    LabelImage classLabeling = maxLabeling;
    size_t iter = 0;
    for (; (numIter > 0) ? (iter < numIter) : (lastEnergy - energy >= m_eps || iter == 0); ++iter)
    {
        lastEnergy = energy;

        // Update superpixels using the latest class labeling
        clusterer.run(m_numClusters, m_numClasses, m_color, classLabeling);
        spLabeling = clusterer.clustership();

        // Update class labeling using the latest superpixels
        optimizer.run(m_color, spLabeling, m_numClusters);
        classLabeling = optimizer.labeling();
        auto clusters = Clusterer<EnergyFun>::computeClusters(spLabeling, m_color, classLabeling, m_numClusters,
                                                              m_numClasses, m_energy);

        energy = m_energy.giveEnergy(classLabeling, m_color, spLabeling, clusters);
        result.energy.push_back(energy);
        result.labelings.push_back(classLabeling);
        result.superpixels.push_back(spLabeling);
    }
    result.numIter = iter;

    return result;
}

template<typename EnergyFun, template<typename> class Optimizer>
float InferenceIterator<EnergyFun, Optimizer>::computeInitialEnergy(LabelImage const& labeling) const
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
