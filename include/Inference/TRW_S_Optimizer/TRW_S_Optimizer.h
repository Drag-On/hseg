//
// Created by jan on 30.11.16.
//

#ifndef HSEG_TRW_S_OPTIMIZER_H
#define HSEG_TRW_S_OPTIMIZER_H

#include <MRFEnergy.h>
#include <typeGeneral.h>
#include <Image/Image.h>
#include <helper/coordinate_helper.h>

/**
 * Opimizes the energy function for the class labeling
 * @tparam EnergyFun Energy function to use
 */
template<typename EnergyFun>
class TRW_S_Optimizer
{
public:
    /**
     * Constructor
     * @param energy Energy function to optimize for l. The reference needs to stay valid as long as the optimizer exists
     */
    TRW_S_Optimizer(EnergyFun const& energy) noexcept;

    /**
     * Runs the optimization
     * @param img Color image
     * @param sp Superpixel labeling
     * @param numSP Amount of superpixels
     * @details This can be called multiples times on the same object. It will then warm-start the algorithm and
     *          initialize it with the previous result.
     */
    template<typename T>
    void run(ColorImage<T> const& img, LabelImage const& sp, Label numSP, std::vector<Cluster> const& clusters);

    /**
     * @return The computed labeling
     */
    LabelImage const& labeling() const;

private:
    EnergyFun m_energy;
    LabelImage m_labeling;
};

template<typename EnergyFun>
TRW_S_Optimizer<EnergyFun>::TRW_S_Optimizer(EnergyFun const& energy) noexcept
        : m_energy(energy)
{
}

template<typename EnergyFun>
template<typename T>
void TRW_S_Optimizer<EnergyFun>::run(ColorImage<T> const& img, LabelImage const& sp, Label numSP, std::vector<Cluster> const& clusters)
{
    SiteId numPx = img.pixels();
    SiteId numNodes = numPx + numSP;
    Label numClasses = m_energy.numClasses();

    // Set up the graph
    TypeGeneral::GlobalSize globalSize;
    MRFEnergy<TypeGeneral> mrfEnergy(globalSize);

    std::vector<MRFEnergy<TypeGeneral>::NodeId> nodeIds;
    nodeIds.reserve(numNodes);

    // Unaries
    for (SiteId i = 0; i < numNodes; ++i)
    {
        // Unary confidences
        std::vector<TypeGeneral::REAL> confidences(numClasses, 0.f);
        if (i < numPx) // Only nodes that represent pixels have unaries.
            for (Label l = 0; l < numClasses; ++l)
                confidences[l] = m_energy.unaryCost(i, l) + m_energy.classData(i, clusters[sp.atSite(i)].label);
        auto id = mrfEnergy.AddNode(TypeGeneral::LocalSize(numClasses), TypeGeneral::NodeData(confidences.data()));
        nodeIds.push_back(id);
    }

    // Pairwise
    for (SiteId i = 0; i < numPx; ++i)
    {
        auto coords = helper::coord::siteTo2DCoordinate(i, img.width());

        if (m_energy.weights().hasPairwiseWeight())
        {
            // Set up pixel neighbor connections
            decltype(coords) coordsR = {coords.x() + 1, coords.y()};
            decltype(coords) coordsD = {coords.x(), coords.y() + 1};
            if (coordsR.x() < img.width())
            {
                SiteId siteR = helper::coord::coordinateToSite(coordsR.x(), coordsR.y(), img.width());
                std::vector<TypeGeneral::REAL> costMat(numClasses * numClasses, 0.f);
                Cost pxCost = m_energy.pairwisePixelWeight(img, i, siteR);
                for(Label l1 = 0; l1 < numClasses; ++l1)
                {
                    for(Label l2 = 0; l2 < l1; ++l2)
                    {
                        Cost cost = m_energy.pairwiseClassWeight(l1, l2);
                        costMat[l1 + l2 * numClasses] = costMat[l2 + l1 * numClasses] = pxCost * cost;
                    }
                }
                TypeGeneral::EdgeData edgeData(TypeGeneral::GENERAL, costMat.data());
                mrfEnergy.AddEdge(nodeIds[i], nodeIds[siteR], edgeData);
            }
            if (coordsD.y() < img.height())
            {
                SiteId siteD = helper::coord::coordinateToSite(coordsD.x(), coordsD.y(), img.width());
                std::vector<TypeGeneral::REAL> costMat(numClasses * numClasses, 0.f);
                Cost pxCost = m_energy.pairwisePixelWeight(img, i, siteD);
                for(Label l1 = 0; l1 < numClasses; ++l1)
                {
                    for(Label l2 = 0; l2 < l1; ++l2)
                    {
                        Cost cost = m_energy.pairwiseClassWeight(l1, l2);
                        costMat[l1 + l2 * numClasses] = costMat[l2 + l1 * numClasses] = pxCost * cost;
                    }
                }
                TypeGeneral::EdgeData edgeData(TypeGeneral::GENERAL, costMat.data());
                mrfEnergy.AddEdge(nodeIds[i], nodeIds[siteD], edgeData);
            }
        }

        // Set up connection to auxiliary nodes
        SiteId auxSite = sp.atSite(i) + numPx;
        std::vector<TypeGeneral::REAL> costMat(numClasses * numClasses, 0.f);
        for(Label l1 = 0; l1 < numClasses; ++l1)
        {
            for(Label l2 = 0; l2 < l1; ++l2)
            {
                Cost cost = m_energy.classDistance(l1, l2);
                costMat[l1 + l2 * numClasses] = costMat[l2 + l1 * numClasses] = cost;
            }
        }
        TypeGeneral::EdgeData edgeData(TypeGeneral::GENERAL, costMat.data());
        mrfEnergy.AddEdge(nodeIds[i], nodeIds[auxSite], edgeData);
    }

    // Do the actual minimization
    MRFEnergy<TypeGeneral>::Options options;
    options.m_eps = 0.0f;
    MRFEnergy<TypeGeneral>::REAL lowerBound = 0, energy = 0;
    mrfEnergy.Minimize_TRW_S(options, lowerBound, energy);

    // Copy over result
    m_labeling = LabelImage(img.width(), img.height());
    for (SiteId i = 0; i < numPx; ++i)
        m_labeling.atSite(i) = mrfEnergy.GetSolution(nodeIds[i]);

}

template<typename EnergyFun>
LabelImage const& TRW_S_Optimizer<EnergyFun>::labeling() const
{
    return m_labeling;
}

#endif //HSEG_TRW_S_OPTIMIZER_H
