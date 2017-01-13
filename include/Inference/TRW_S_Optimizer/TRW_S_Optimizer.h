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
     * @param energy Energy function to optimize for l. The pointer needs to stay valid as long as the optimizer exists
     */
    TRW_S_Optimizer(EnergyFun const* energy) noexcept;

    /**
     * Runs the optimization
     * @param img Feature image
     * @details This can be called multiples times on the same object. It will then warm-start the algorithm and
     *          initialize it with the previous result.
     */
    void run(FeatureImage const& img);

    /**
     * @return The computed labeling
     */
    LabelImage const& labeling() const;

private:
    EnergyFun const* m_pEnergy;
    LabelImage m_labeling;
};

template<typename EnergyFun>
TRW_S_Optimizer<EnergyFun>::TRW_S_Optimizer(EnergyFun const* energy) noexcept
        : m_pEnergy(energy)
{
}

template<typename EnergyFun>
void TRW_S_Optimizer<EnergyFun>::run(FeatureImage const& img)
{
    SiteId numPx = img.width() * img.height();
    SiteId numNodes = numPx;
    Label numClasses = m_pEnergy->numClasses();

    // Set up the graph
    TypeGeneral::GlobalSize globalSize;
    MRFEnergy<TypeGeneral> mrfEnergy(globalSize);

    std::vector<MRFEnergy<TypeGeneral>::NodeId> nodeIds;
    nodeIds.reserve(numNodes);

    // Unaries
    for (SiteId i = 0; i < numNodes; ++i)
    {
        // Unary confidences
        Feature const& f = img.atSite(i);
        std::vector<TypeGeneral::REAL> confidences(numClasses, 0.f);
        for (Label l = 0; l < numClasses; ++l)
            confidences[l] = m_pEnergy->unaryCost(f, l);
        auto id = mrfEnergy.AddNode(TypeGeneral::LocalSize(numClasses), TypeGeneral::NodeData(confidences.data()));
        nodeIds.push_back(id);
    }

    // Pairwise
    for (SiteId i = 0; i < numPx; ++i)
    {
        auto coords = helper::coord::siteTo2DCoordinate(i, img.width());
        Feature const& f = img.atSite(i);

        // Set up pixel neighbor connections
        decltype(coords) coordsR = {coords.x() + 1, coords.y()};
        decltype(coords) coordsD = {coords.x(), coords.y() + 1};
        if (coordsR.x() < img.width())
        {
            SiteId siteR = helper::coord::coordinateToSite(coordsR.x(), coordsR.y(), img.width());
            Feature const& fR = img.atSite(siteR);
            std::vector<TypeGeneral::REAL> costMat(numClasses * numClasses, 0.f);
            for(Label l1 = 0; l1 < numClasses; ++l1)
            {
                for(Label l2 = 0; l2 < l1; ++l2)
                {
                    Cost cost = m_pEnergy->pairwise(f, fR, l1, l2);
                    costMat[l1 + l2 * numClasses] = costMat[l2 + l1 * numClasses] = cost;
                }
            }
            TypeGeneral::EdgeData edgeData(TypeGeneral::GENERAL, costMat.data());
            mrfEnergy.AddEdge(nodeIds[i], nodeIds[siteR], edgeData);
        }
        if (coordsD.y() < img.height())
        {
            SiteId siteD = helper::coord::coordinateToSite(coordsD.x(), coordsD.y(), img.width());
            Feature const& fD = img.atSite(siteD);
            std::vector<TypeGeneral::REAL> costMat(numClasses * numClasses, 0.f);
            for(Label l1 = 0; l1 < numClasses; ++l1)
            {
                for(Label l2 = 0; l2 < l1; ++l2)
                {
                    Cost cost = m_pEnergy->pairwise(f, fD, l1, l2);
                    costMat[l1 + l2 * numClasses] = costMat[l2 + l1 * numClasses] = cost;
                }
            }
            TypeGeneral::EdgeData edgeData(TypeGeneral::GENERAL, costMat.data());
            mrfEnergy.AddEdge(nodeIds[i], nodeIds[siteD], edgeData);
        }

        // Set up connection to auxiliary nodes
//        SiteId auxSite = sp.atSite(i) + numPx;
//        std::vector<TypeGeneral::REAL> costMat(numClasses * numClasses, 0.f);
//        for(Label l1 = 0; l1 < numClasses; ++l1)
//        {
//            for(Label l2 = 0; l2 < l1; ++l2)
//            {
//                Cost cost = m_energy.classDistance(l1, l2);
//                costMat[l1 + l2 * numClasses] = costMat[l2 + l1 * numClasses] = cost;
//            }
//        }
//        TypeGeneral::EdgeData edgeData(TypeGeneral::GENERAL, costMat.data());
//        mrfEnergy.AddEdge(nodeIds[i], nodeIds[auxSite], edgeData);
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
