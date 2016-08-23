//
// Created by jan on 20.08.16.
//

#ifndef HSEG_GRAPHOPTIMIZER_H
#define HSEG_GRAPHOPTIMIZER_H

#include <Energy/UnaryFile.h>
#include <Image/Image.h>
#include <Energy/EnergyFunction.h>
#include "GCoptimization.h"

/**
 * Opimizes the energy function for the class labeling
 */
class GraphOptimizer
{
public:
    /**
     * Constructor
     * @param energy Energy function to optimize for l
     */
    GraphOptimizer(EnergyFunction const& energy) noexcept;

    /**
     * Runs the optimization
     * @param img Color image
     * @param sp Superpixel labeling
     * @param numSP Amount of superpixels
     * @details This can be called multiples times on the same object. It will then warm-start the algorithm and
     *          initialize it with the previous result.
     */
    template<typename T>
    void run(ColorImage<T> const& img, LabelImage const& sp, size_t numSP);

    /**
     * @return The computed labeling
     */
    LabelImage const& labeling() const;

private:
    EnergyFunction m_energy;
    LabelImage m_labeling;

    template<typename T>
    class PairwiseCost : public GCoptimization::SmoothCostFunctor
    {
    private:
        EnergyFunction const& m_energy;
        ColorImage<T> const& m_color;
    public:
        using EnergyTermType = GCoptimization::EnergyTermType;
        using SiteID = GCoptimization::SiteID;
        using LabelID = GCoptimization::LabelID;

        PairwiseCost(EnergyFunction const& energy, ColorImage<T> const& color);

        EnergyTermType compute(SiteID s1, SiteID s2, LabelID l1, LabelID l2) override;
    };
};

template<typename T>
void GraphOptimizer::run(ColorImage<T> const& img, LabelImage const& sp, size_t numSP)
{
    size_t numPx = img.pixels();
    size_t numNodes = numPx + numSP;

    // Setup graph
    GCoptimizationGeneralGraph graph(numNodes, m_energy.numClasses());
    for (size_t i = 0; i < numPx; ++i)
    {
        auto coords = helper::coord::siteTo2DCoordinate(i, img.width());

        // Set up pixel neighbor connections
        decltype(coords) coordsR = {coords.x() + 1, coords.y()};
        decltype(coords) coordsD = {coords.x(), coords.y() + 1};
        if (coordsR.x() < img.width())
        {
            size_t siteR = helper::coord::coordinateToSite(coordsR.x(), coordsR.y(), img.width());
            graph.setNeighbors(i, siteR, 1);
        }
        if (coordsD.y() < img.height())
        {
            size_t siteD = helper::coord::coordinateToSite(coordsD.x(), coordsD.y(), img.width());
            graph.setNeighbors(i, siteD, 1);
        }

        // Set up connection to auxiliary nodes
        size_t auxSite = sp.atSite(i) + numPx;
        graph.setNeighbors(i, auxSite, 1);

        // Set up unary cost
        for (Label l = 0; l < m_energy.numClasses(); ++l)
            graph.setDataCost(i, l, m_energy.unaryCost(i, l));
    }

    // Set up pairwise cost
    PairwiseCost<T> pairwiseCost(m_energy, img);
    graph.setSmoothCostFunctor(&pairwiseCost);

    // Warm start from previous labeling if available
    if (m_labeling.width() == img.width() && m_labeling.height() == img.height())
    {
        for (size_t s = 0; s < m_labeling.pixels(); ++s)
            graph.setLabel(s, m_labeling.atSite(s));
    }

    // Do alpha-expansion
    graph.expansion();

    // Copy over result
    m_labeling = LabelImage(img.width(), img.height());
    for (size_t i = 0; i < numPx; ++i)
        m_labeling.atSite(i) = graph.whatLabel(i);
}

template<typename T>
GraphOptimizer::PairwiseCost<T>::PairwiseCost(EnergyFunction const& energy, ColorImage<T> const& color)
        : m_energy(energy),
          m_color(color)
{
}

template<typename T>
typename GraphOptimizer::PairwiseCost<T>::EnergyTermType
GraphOptimizer::PairwiseCost<T>::compute(GraphOptimizer::PairwiseCost<T>::SiteID s1,
                                         GraphOptimizer::PairwiseCost<T>::SiteID s2,
                                         GraphOptimizer::PairwiseCost<T>::LabelID l1,
                                         GraphOptimizer::PairwiseCost<T>::LabelID l2)
{
    // If the labels are identical it's always zero
    if(l1 == l2)
        return 0;

    // If both sites are normal nodes just compute the normal pairwise
    if (static_cast<Label>(s1) < m_color.pixels() && static_cast<Label>(s2) < m_color.pixels())
        return m_energy.pairwisePixelWeight(m_color, s1, s2) * m_energy.pairwiseClassWeight(l1, l2);

    // Otherwise one of the nodes is an auxilliary node, therefore apply the class weight
    return m_energy.classDistance(l1, l2);
}

#endif //HSEG_GRAPHOPTIMIZER_H
