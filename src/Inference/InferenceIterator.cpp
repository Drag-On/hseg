//
// Created by jan on 28.08.16.
//

#include <Inference/GraphOptimizer/GraphOptimizer.h>
#include <Inference/k-prototypes/Clusterer.h>
#include "Inference/InferenceIterator.h"

InferenceIterator::InferenceIterator(EnergyFunction const& e, size_t numClusters, size_t numClasses,
                                     CieLabImage const& color)
        : m_energy(e),
          m_numClusters(numClusters),
          m_numClasses(numClasses),
          m_color(color)
{
}

InferenceResult InferenceIterator::run(size_t numIter)
{
    InferenceResult result(m_energy);
    Clusterer& clusterer = result.clusterer;
    GraphOptimizer& optimizer = result.optimizer;

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

InferenceResultDetails InferenceIterator::runDetailed(size_t numIter)
{
    Clusterer clusterer(m_energy);
    GraphOptimizer optimizer(m_energy);

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

        float energy = m_energy.giveEnergy(classLabeling, m_color, spLabeling, clusterer.clusters());
        result.energy.push_back(energy);
        result.labelings.push_back(classLabeling);
        result.superpixels.push_back(spLabeling);
    }
    result.numIter = numIter;

    return result;
}

float InferenceIterator::computeInitialEnergy(LabelImage const& labeling) const
{
    LabelImage fakeSpLabeling(m_color.width(), m_color.height());
    Cluster c(m_numClasses);
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
