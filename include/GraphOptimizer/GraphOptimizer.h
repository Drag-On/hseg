//
// Created by jan on 20.08.16.
//

#ifndef HSEG_GRAPHOPTIMIZER_H
#define HSEG_GRAPHOPTIMIZER_H

#include <UnaryFile.h>
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
        decltype(coords) coordsR = {coords.first + 1, coords.second};
        decltype(coords) coordsD = {coords.first, coords.second + 1};
        if (coordsR.first < img.width())
        {
            size_t siteR = helper::coord::coordinateToSite(coordsR.first, coordsR.second, img.width());
            graph.setNeighbors(i, siteR, m_energy.pairwiseWeight(img, i, siteR));
        }
        if (coordsD.second < img.height())
        {
            size_t siteD = helper::coord::coordinateToSite(coordsD.first, coordsD.second, img.width());
            graph.setNeighbors(i, siteD, m_energy.pairwiseWeight(img, i, siteD));
        }

        // Set up connection to auxiliary nodes
        size_t auxSite = sp.atSite(i) + numPx;
        graph.setNeighbors(i, auxSite, m_energy.classWeight());

        // Set up unary cost
        for (Label l = 0; l < m_energy.numClasses(); ++l)
            graph.setDataCost(i, l, m_energy.unaryCost(i, l));
    }

    // Set up pairwise cost
    for (size_t l1 = 0; l1 < m_energy.numClasses(); ++l1)
    {
        for (size_t l2 = 0; l2 < m_energy.numClasses(); ++l2)
        {
            float cost = l1 == l2 ? 0 : 1;
            graph.setSmoothCost(l1, l2, cost);
            graph.setSmoothCost(l2, l1, cost);
        }
    }

    // Do alpha-expansion
    graph.expansion();

    // Copy over result
    m_labeling = LabelImage(img.width(), img.height());
    for (size_t i = 0; i < numPx; ++i)
        m_labeling.atSite(i) = graph.whatLabel(i);
}

#endif //HSEG_GRAPHOPTIMIZER_H
