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
    InferenceIterator(EnergyFun e, Label numClusters, Label numClasses, CieLabImage const& color);

    /**
     * Does the actual inference
     * @param numIter Amount of iterations to do. If 0, run until convergence.
     * @return Resulting labeling and segmentation
     */
    InferenceResult run(uint32_t numIter = 0);

    /**
     * Does inference and saves detailed results
     * @param numIter Amount of iterations to do. If 0, run until convergence.
     * @return Detailed results
     */
    InferenceResultDetails runDetailed(uint32_t numIter = 0);

private:
    EnergyFun m_energy;
    Label m_numClusters;
    Label m_numClasses;
    CieLabImage const& m_color;
    float const m_eps = 1e3f;

    Cost computeInitialEnergy(LabelImage const& labeling) const;
};

template<typename EnergyFun, template<typename> class Optimizer>
InferenceIterator<EnergyFun, Optimizer>::InferenceIterator(EnergyFun e, Label numClusters, Label numClasses,
                                                CieLabImage const& color)
        : m_energy(e),
          m_numClusters(numClusters),
          m_numClasses(numClasses),
          m_color(color)
{
}

template<typename EnergyFun, template<typename> class Optimizer>
InferenceResult InferenceIterator<EnergyFun, Optimizer>::run(uint32_t numIter)
{
    InferenceResult result;

    Cost energy = std::numeric_limits<Cost>::max();
    Cost lastEnergy = energy;
    //result.labeling = m_energy.unaryFile().maxLabeling();
    result.labeling = LabelImage(m_color.width(), m_color.height()); // The labeling will be empty (all zeros)

    Clusterer<EnergyFun> clusterer(m_energy, m_color, result.labeling, m_numClusters);
    Optimizer<EnergyFun> optimizer(m_energy);

    for (uint32_t iter = 0; (numIter > 0) ? (iter < numIter) : (lastEnergy - energy >= m_eps || iter == 0); ++iter)
    {
        lastEnergy = energy;

        // Update superpixels using the latest class labeling
        clusterer.run(result.labeling);
        result.superpixels = clusterer.clustership();

        // Update class labeling using the latest superpixels
        optimizer.run(m_color, result.superpixels, m_numClusters);
        result.labeling = optimizer.labeling();

        if(numIter == 0)
            energy = m_energy.giveEnergy(result.labeling, m_color, result.superpixels, clusterer.clusters());
    }

    return result;
}

template<typename EnergyFun, template<typename> class Optimizer>
InferenceResultDetails InferenceIterator<EnergyFun, Optimizer>::runDetailed(uint32_t numIter)
{
    InferenceResultDetails result;

    //LabelImage maxLabeling = m_energy.unaryFile().maxLabeling();
    LabelImage maxLabeling = LabelImage(m_color.width(), m_color.height()); // The labeling will be empty (all zeros)
    Cost initialEnergy = computeInitialEnergy(maxLabeling);
    Cost lastEnergy = initialEnergy;
    Cost energy = initialEnergy;
    result.energy.push_back(initialEnergy);

    Clusterer<EnergyFun> clusterer(m_energy, m_color, maxLabeling, m_numClusters);
    Optimizer<EnergyFun> optimizer(m_energy);

    LabelImage spLabeling;
    LabelImage classLabeling = maxLabeling;
    uint32_t iter = 0;
    for (; (numIter > 0) ? (iter < numIter) : (lastEnergy - energy >= m_eps || iter == 0); ++iter)
    {
        lastEnergy = energy;

        // Update superpixels using the latest class labeling
        clusterer.run(classLabeling);
        spLabeling = clusterer.clustership();

        // Update class labeling using the latest superpixels
        optimizer.run(m_color, spLabeling, m_numClusters);
        classLabeling = optimizer.labeling();

        energy = m_energy.giveEnergy(classLabeling, m_color, spLabeling, clusterer.clusters());
        result.energy.push_back(energy);
        result.labelings.push_back(classLabeling);
        result.superpixels.push_back(spLabeling);
    }
    result.numIter = iter;

    return result;
}

template<typename EnergyFun, template<typename> class Optimizer>
Cost InferenceIterator<EnergyFun, Optimizer>::computeInitialEnergy(LabelImage const& labeling) const
{
    LabelImage fakeSpLabeling(m_color.width(), m_color.height());
    Cluster c(&m_energy);
    for (SiteId i = 0; i < m_color.pixels(); ++i)
    {
        c.accumFeature += Feature(m_color, i);
        c.labelFrequencies[labeling.atSite(i)]++;
    }
    c.size = m_color.pixels();
    c.updateMean();
    c.updateLabel();
    std::vector<Cluster> fakeClusters(1, c);

    Cost energy = m_energy.giveEnergy(labeling, m_color, fakeSpLabeling, fakeClusters);
    return energy;
}


#endif //HSEG_INFERENCEITERATOR_H
