//
// Created by jan on 28.08.16.
//

#ifndef HSEG_INFERENCEITERATOR_H
#define HSEG_INFERENCEITERATOR_H

#include <Energy/EnergyFunction.h>
#include <Image/FeatureImage.h>
#include <Inference/TRW_S_Optimizer/TRW_S_Optimizer.h>
#include "InferenceResult.h"
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
     * @param pImg Color image
     */
    InferenceIterator(EnergyFun const* e, FeatureImage const* pImg);

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
    EnergyFun const* m_pEnergy;
    FeatureImage const* m_pImg;
    float const m_eps = 1e3f;

    Cost computeInitialEnergy(LabelImage const& labeling) const;
};

template<typename EnergyFun>
InferenceIterator<EnergyFun>::InferenceIterator(EnergyFun const* e, FeatureImage const* pImg)
        : m_pEnergy(e),
          m_pImg(pImg)
{
}

template<typename EnergyFun>
InferenceResult InferenceIterator<EnergyFun>::run(uint32_t numIter)
{
    InferenceResult result;

    Cost energy = std::numeric_limits<Cost>::max();
    Cost lastEnergy = energy;
    //result.labeling = m_energy.unaryFile().maxLabeling();
    result.labeling = LabelImage(m_pImg->width(), m_pImg->height()); // The labeling will be empty (all zeros)

    TRW_S_Optimizer<EnergyFun> optimizer(m_pEnergy);
    optimizer.run(*m_pImg);
    result.labeling = optimizer.labeling();

    // TODO: Optimize

//    for (result.numIter = 0; (numIter > 0) ? (result.numIter < numIter) : (lastEnergy - energy >= m_eps || result.numIter == 0); ++result.numIter)
//    {
//        lastEnergy = energy;
//
//        // Update superpixels using the latest class labeling
//        clusterer.run(result.labeling);
//        result.superpixels = clusterer.clustership();
//        result.clusters = clusterer.clusters();
//
//        // Update class labeling using the latest superpixels
//        optimizer.run(m_color, result.superpixels, m_numClusters);
//        result.labeling = optimizer.labeling();
//
//        if(numIter == 0)
//        {
//            energy = m_energy.giveEnergy(result.labeling, m_color, result.superpixels, clusterer.clusters());
//            // std::cout <<  result.numIter << ": " << energy << " | " << lastEnergy - energy << " >= " << m_eps << std::endl;
//        }
//    }

    return result;
}

template<typename EnergyFun>
InferenceResultDetails InferenceIterator<EnergyFun>::runDetailed(uint32_t numIter)
{
    InferenceResultDetails result;

    //LabelImage maxLabeling = m_energy.unaryFile().maxLabeling();
    LabelImage maxLabeling = LabelImage(m_pImg->width(), m_pImg->height()); // The labeling will be empty (all zeros)
    Cost initialEnergy = computeInitialEnergy(maxLabeling);
    Cost lastEnergy = initialEnergy;
    Cost energy = initialEnergy;
    result.energy.push_back(initialEnergy);

    TRW_S_Optimizer<EnergyFun> optimizer(m_pEnergy);
    optimizer.run(*m_pImg);
    result.labelings.push_back(optimizer.labeling());
    result.marginals.push_back(optimizer.marginals());
    energy = m_pEnergy->giveEnergy(*m_pImg, optimizer.labeling());
    result.energy.push_back(energy);
    result.numIter = 1;

//    LabelImage spLabeling;
//    LabelImage classLabeling = maxLabeling;
//    uint32_t iter = 0;
//
//
//
//    for (; (numIter > 0) ? (iter < numIter) : (lastEnergy - energy >= m_eps || iter == 0); ++iter)
//    {
//        lastEnergy = energy;
//
//        // Update superpixels using the latest class labeling
//        clusterer.run(classLabeling);
//        spLabeling = clusterer.clustership();
//
//        // Update class labeling using the latest superpixels
//        optimizer.run(m_color, spLabeling, m_numClusters);
//        classLabeling = optimizer.labeling();
//
//        energy = m_energy.giveEnergy(classLabeling, m_color, spLabeling, clusterer.clusters());
//        result.energy.push_back(energy);
//        result.labelings.push_back(classLabeling);
//        result.superpixels.push_back(spLabeling);
//    }
//    result.numIter = iter;

    return result;
}

template<typename EnergyFun>
Cost InferenceIterator<EnergyFun>::computeInitialEnergy(LabelImage const& labeling) const
{
    LabelImage fakeSpLabeling(m_pImg->width(), m_pImg->height());
    std::vector<Feature> features;

    Cost energy = m_pEnergy->giveEnergy(*m_pImg, labeling);
    return energy;
}


#endif //HSEG_INFERENCEITERATOR_H
