//
// Created by jan on 20.08.16.
//

#ifndef HSEG_GRAPHOPTIMIZER_H
#define HSEG_GRAPHOPTIMIZER_H

#include <Energy/UnaryFile.h>
#include <Image/Image.h>
#include <Energy/EnergyFunction.h>
#include <boost/unordered_map.hpp>
//#include <helper/hash_helper.h>
#include "GCoptimization.h"
#include "typedefs.h"

/**
 * Opimizes the energy function for the class labeling
 */
template<typename EnergyFun>
class GraphOptimizer
{
public:
    /**
     * Constructor
     * @param energy Energy function to optimize for l. The reference needs to stay valid as long as the optimizer exists
     */
    GraphOptimizer(EnergyFun energy) noexcept;

    /**
     * Runs the optimization
     * @param img Color image
     * @param sp Superpixel labeling
     * @param numSP Amount of superpixels
     * @details This can be called multiples times on the same object. It will then warm-start the algorithm and
     *          initialize it with the previous result.
     */
    template<typename T>
    void run(ColorImage<T> const& img, LabelImage const& sp, Label numSP);

    /**
     * @return The computed labeling
     */
    LabelImage const& labeling() const;

private:
    EnergyFun m_energy;
    LabelImage m_labeling;
    static constexpr GCoptimization::EnergyTermType s_constFactor = 1000; // Used because the optimizer uses integer energies

    template<typename T>
    class PairwiseCost : public GCoptimization::SmoothCostFunctor
    {
    public:
        using EnergyTermType = GCoptimization::EnergyTermType;
        using SiteID = GCoptimization::SiteID;
        using LabelID = GCoptimization::LabelID;
        using PixelPairType = std::pair<SiteID, SiteID>;

        EnergyFun const* m_pEnergy;
        ColorImage<T> const& m_color;

        PairwiseCost(EnergyFun const& energy, ColorImage<T> const& color);

        EnergyTermType compute(SiteID s1, SiteID s2, LabelID l1, LabelID l2) override;
    };
};

template<typename EnergyFun>
GraphOptimizer<EnergyFun>::GraphOptimizer(EnergyFun energy) noexcept
        : m_energy(energy)
{
}

template<typename EnergyFun>
inline LabelImage const& GraphOptimizer<EnergyFun>::labeling() const
{
    return m_labeling;
}


template<typename EnergyFun>
template<typename T>
void GraphOptimizer<EnergyFun>::run(ColorImage<T> const& img, LabelImage const& sp, Label numSP)
{
    SiteId numPx = img.pixels();
    SiteId numNodes = numPx + numSP;

    // Setup graph
    GCoptimizationGeneralGraph graph(numNodes, m_energy.numClasses());
    std::vector<std::vector<SiteId>> clusterAssociation(numSP);
    for (SiteId i = 0; i < numPx; ++i)
    {
        auto coords = helper::coord::siteTo2DCoordinate(i, img.width());

        if(m_energy.weights().hasPairwiseWeight())
        {
            // Set up pixel neighbor connections
            decltype(coords) coordsR = {static_cast<Coord>(coords.x() + 1), coords.y()};
            decltype(coords) coordsD = {coords.x(), static_cast<Coord>(coords.y() +  1)};
            if (coordsR.x() < img.width())
            {
                SiteId siteR = helper::coord::coordinateToSite(coordsR.x(), coordsR.y(), img.width());
                graph.setNeighbors(i, siteR, 1);
            }
            if (coordsD.y() < img.height())
            {
                SiteId siteD = helper::coord::coordinateToSite(coordsD.x(), coordsD.y(), img.width());
                graph.setNeighbors(i, siteD, 1);
            }
        }

        // Set up connection to auxiliary nodes
        SiteId auxSite = sp.atSite(i) + numPx;
        graph.setNeighbors(i, auxSite, 1);

        // Set up unary cost
        for (Label l = 0; l < m_energy.numClasses(); ++l)
        {
            Cost cost = m_energy.unaryCost(i, l);
            graph.setDataCost(i, l, static_cast<GCoptimization::EnergyTermType>(std::round(cost * s_constFactor)));
        }
        // Remember the cluster this pixel is assigned to
        clusterAssociation[sp.atSite(i)].push_back(i);
    }
    // Set up unary cost for the superpixel nodes
    for(Label sp = 0; sp < numSP; ++sp)
    {
        for(Label l = 0; l < m_energy.numClasses(); ++l)
        {
            Cost cost = 0.f;
            for(auto const& i : clusterAssociation[sp])
                cost += m_energy.classData(i, l);
            graph.setDataCost(numPx + sp, l, static_cast<GCoptimization::EnergyTermType>(std::round(cost * s_constFactor)));
        }
    }

    // Set up pairwise cost
    PairwiseCost<T> pairwiseCost(m_energy, img);
    graph.setSmoothCostFunctor(&pairwiseCost);

    // Warm start from previous labeling if available
    if (m_labeling.width() == img.width() && m_labeling.height() == img.height())
    {
        for (SiteId s = 0; s < m_labeling.pixels(); ++s)
            graph.setLabel(s, m_labeling.atSite(s));
    }

    // Do alpha-beta-swap
    try
    {
        graph.swap();
    }
    catch(GCException e)
    {
        e.Report();
    }

    // Copy over result
    m_labeling = LabelImage(img.width(), img.height());
    for (SiteId i = 0; i < numPx; ++i)
        m_labeling.atSite(i) = graph.whatLabel(i);
}

template<typename EnergyFun>
template<typename T>
GraphOptimizer<EnergyFun>::PairwiseCost<T>::PairwiseCost(EnergyFun const& energy, ColorImage<T> const& color)
        : m_pEnergy(&energy),
          m_color(color)
{
}

template<typename EnergyFun>
template<typename T>
typename GraphOptimizer<EnergyFun>::template PairwiseCost<T>::EnergyTermType
GraphOptimizer<EnergyFun>::PairwiseCost<T>::compute(GraphOptimizer<EnergyFun>::PairwiseCost<T>::SiteID s1,
                                                    GraphOptimizer<EnergyFun>::PairwiseCost<T>::SiteID s2,
                                                    GraphOptimizer<EnergyFun>::PairwiseCost<T>::LabelID l1,
                                                    GraphOptimizer<EnergyFun>::PairwiseCost<T>::LabelID l2)
{
    using EnergyTermType = GraphOptimizer<EnergyFun>::PairwiseCost<T>::EnergyTermType;
    // If the labels are identical it's always zero
    if (l1 == l2)
        return 0;

    // Cached entries always have the lower index first
    if (s2 < s1)
        std::swap(s1, s2);

    // If both sites are normal nodes just compute the normal pairwise
    if (static_cast<Label>(s1) < m_color.pixels() && static_cast<Label>(s2) < m_color.pixels())
    {
        std::pair<SiteID, SiteID> pair{s1, s2};
        Cost pxEnergy = m_pEnergy->pairwisePixelWeight(m_color, s1, s2);
        Cost classWeight = m_pEnergy->pairwiseClassWeight(l1, l2);
        return static_cast<EnergyTermType>(std::round(pxEnergy * classWeight * s_constFactor));
    }
    else // Otherwise one of the nodes is an auxilliary node, therefore apply the class weight
        return static_cast<EnergyTermType>(std::round(m_pEnergy->classDistance(l1, l2) * s_constFactor));
}

#endif //HSEG_GRAPHOPTIMIZER_H
