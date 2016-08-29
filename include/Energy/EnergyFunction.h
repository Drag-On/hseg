//
// Created by jan on 20.08.16.
//

#ifndef HSEG_ENERGYFUNCTION_H
#define HSEG_ENERGYFUNCTION_H


#include <Energy/UnaryFile.h>
#include <Inference/k-prototypes/Feature.h>
#include <Inference/k-prototypes/Cluster.h>
#include "Weights.h"

/**
 * Provides functionality to compute (partial) energies of the target energy function
 */
class EnergyFunction
{
public:
    /**
     * Constructor
     * @param unaries Unary scores. The reference must stay valid as long as this object persists.
     * @param weights Weights to use. The reference must stay valid as long as this object persists.
     * @param pairwiseSigmaSq Sigma-Square inside of the exponential
     */
    EnergyFunction(UnaryFile const& unaries, Weights const& weights, float pairwiseSigmaSq = 0.05f);

    /**
     * Computes the overall energy
     * @param labeling Class labeling
     * @param img Color image
     * @param sp Superpixel labeling
     * @param clusters List of clusters
     * @return The energy of the given configuration
     */
    template<typename T>
    float giveEnergy(LabelImage const& labeling, ColorImage<T> const& img, LabelImage const& sp,
                     std::vector<Cluster> const& clusters) const;

    /**
     * Computes the unary energy
     * @param labeling Class labeling
     * @return The unary energy of the given configuration
     */
    float giveUnaryEnergy(LabelImage const& labeling) const;

    /**
     * Computes the pairwise energy
     * @param labeling Class labeling
     * @param img Color image
     * @return The pairwise energy of the given configuration
     */
    template<typename T>
    float givePairwiseEnergy(LabelImage const& labeling, ColorImage<T> const& img) const;

    /**
     * Computes the superpixel energy
     * @param labeling Class labeling
     * @param img Color image
     * @param sp Superpixel labeling
     * @param clusters List of clusters
     * @return The superpixel energy of the given configuration
     */
    template<typename T>
    float giveSpEnergy(LabelImage const& labeling, ColorImage<T> const& img, LabelImage const& sp,
                       std::vector<Cluster> const& clusters) const;

    /**
     * @return Amount of classes
     */
    inline Label numClasses() const
    {
        return static_cast<Label>(m_unaryScores.classes());
    }

    /**
     * Computes the cost of a unary-class label combination
     * @param i Pixel to compute unary for
     * @param l Class label to compute score for
     * @return The cost of assigning class label \p l to pixel \p i
     */
    virtual float unaryCost(size_t i, Label l) const;

    /**
     * Computes the partial cost of a pairwise connection as given by the color of the pixels
     * @param img Color image
     * @param i First pixel id
     * @param j Second pixel id
     * @return The partial cost
     */
    template<typename T>
    float pairwisePixelWeight(ColorImage<T> const& img, size_t i, size_t j) const;

    /**
     * Computes the partial cost of a pairwise connection as given by the labels of the pixels
     * @param l1 First label
     * @param l2 Second label
     * @return The partial cost
     */
    inline float pairwiseClassWeight(Label l1, Label l2) const
    {
        return m_weights.pairwise(l1, l2);
    }

    /**
     * Computes the feature distance between two features
     * @param feature The first feature
     * @param feature2 The second feature
     * @return The feature distance
     */
    float featureDistance(Feature const& feature, Feature const& feature2) const;

    /**
     * Computes the class label distance between two class labels
     * @param l1 First class label
     * @param l2 Second class label
     * @return The class label distance
     */
    inline float classDistance(Label l1, Label l2) const
    {
        if (l1 == l2)
            return 0;
        else
            return m_weights.classWeight();
    }

    /**
     * Computes the pixel-to-cluster distance
     * @param fPx Feature of the pixel
     * @param lPx Class label of the pixel
     * @param cl Cluster
     * @return The pixel-to-cluster distance
     */
    inline float pixelToClusterDistance(Feature const& fPx, Label lPx, Cluster const& cl) const
    {
        return featureDistance(fPx, cl.mean) + classDistance(lPx, cl.label);
    }

    /**
     * Simple potts model.
     * @param l1 First label
     * @param l2 Second label
     * @return 0 in case the labels are identical, otherwise 1
     */
    template<typename T>
    inline float simplePotts(T l1, T l2) const;

    /**
     * @return The unary file
     */
    UnaryFile const& unaryFile() const;

private:
    UnaryFile const& m_unaryScores;
    Weights const& m_weights;
    float m_pairWiseSigmaSq;
};

template<typename T>
float EnergyFunction::giveEnergy(LabelImage const& labeling, ColorImage<T> const& img, LabelImage const& sp,
                                 std::vector<Cluster> const& clusters) const
{
    auto unaryEnergy = giveUnaryEnergy(labeling);
    auto pairwiseEnergy = givePairwiseEnergy(labeling, img);
    auto spEnergy = giveSpEnergy(labeling, img, sp, clusters);
    return unaryEnergy + pairwiseEnergy + spEnergy;
}

template<typename T>
float EnergyFunction::givePairwiseEnergy(LabelImage const& labeling, ColorImage<T> const& img) const
{
    float pairwiseEnergy = 0;
    for (size_t x = 0; x < labeling.width() - 1; ++x)
    {
        for (size_t y = 0; y < labeling.height() - 1; ++y)
        {
            Label l = labeling.at(x, y);
            Label lR = labeling.at(x + 1, y);
            if (l != lR)
                pairwiseEnergy += pairwiseClassWeight(l, lR)
                                  * pairwisePixelWeight(img, helper::coord::coordinateToSite(x, y, labeling.width()),
                                                        helper::coord::coordinateToSite(x + 1, y, labeling.width()));
            Label lD = labeling.at(x, y + 1);
            if (l != lD)
                pairwiseEnergy += pairwiseClassWeight(l, lD)
                                  * pairwisePixelWeight(img, helper::coord::coordinateToSite(x, y, labeling.width()),
                                                        helper::coord::coordinateToSite(x, y + 1, labeling.width()));
        }
    }
    return pairwiseEnergy;
}

template<typename T>
float EnergyFunction::pairwisePixelWeight(ColorImage<T> const& img, size_t i, size_t j) const
{
    float rDiff = img.atSite(i, 0) - img.atSite(j, 0);
    float gDiff = img.atSite(i, 1) - img.atSite(j, 1);
    float bDiff = img.atSite(i, 2) - img.atSite(j, 2);
    float colorDiffNormSq = rDiff * rDiff + gDiff * gDiff + bDiff * bDiff;
    float weight = std::exp(-m_pairWiseSigmaSq * colorDiffNormSq);
    return weight;
}

template<typename T>
float EnergyFunction::giveSpEnergy(LabelImage const& labeling, ColorImage<T> const& img, LabelImage const& sp,
                                   std::vector<Cluster> const& clusters) const
{
    float spEnergy = 0;
    for (size_t i = 0; i < labeling.pixels(); ++i)
    {
        assert(clusters.size() > sp.atSite(i));
        spEnergy += pixelToClusterDistance(Feature(img, i), labeling.atSite(i), clusters[sp.atSite(i)]);
    }

    return spEnergy;
}

template<typename T>
inline float EnergyFunction::simplePotts(T l1, T l2) const
{
    if (l1 == l2)
        return 0;
    else
        return 1;
}

#endif //HSEG_ENERGYFUNCTION_H
