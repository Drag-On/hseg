//
// Created by jan on 20.08.16.
//

#ifndef HSEG_ENERGYFUNCTION_H
#define HSEG_ENERGYFUNCTION_H


#include <Energy/UnaryFile.h>
#include <k-prototypes/Feature.h>
#include <Properties.h>
#include "Weights.h"

/**
 * Provides functionality to compute (partial) energies of the target energy function
 */
class EnergyFunction
{
public:
    /**
     * Constructor
     * @param unaries Unary scores
     * @param weights Weights to use
     */
    EnergyFunction(UnaryFile const& unaries, Weights const& weights);

    /**
     * Computes the overall energy
     * @param labeling Class labeling
     * @param img Color image
     * @param sp Superpixel labeling
     * @return The energy of the given configuration
     */
    template<typename T>
    float giveEnergy(LabelImage const& labeling, ColorImage<T> const& img, LabelImage const& sp) const;

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
     * @return The superpixel energy of the given configuration
     */
    template<typename T>
    float giveSpEnergy(LabelImage const& labeling, ColorImage<T> const& img, LabelImage const& sp) const;

    /**
     * @return Amount of classes
     */
    Label numClasses() const;

    /**
     * Computes the cost of a unary-class label combination
     * @param i Pixel to compute unary for
     * @param l Class label to compute score for
     * @return The cost of assigning class label \p l to pixel \p i
     */
    float unaryCost(size_t i, Label l) const;

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
    float pairwiseClassWeight(Label l1, Label l2) const;

    /**
     * @return The cost of non-identical class labels
     */
    float classWeight() const;

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
    float classDistance(Label l1, Label l2) const;

    /**
     * Computes the pixel-to-cluster distance
     * @param fPx Feature of the pixel
     * @param lPx Class label of the pixel
     * @param fCl Feature of the cluster
     * @param lCl Class label of the cluster
     * @return The pixel-to-cluster distance
     */
    float pixelToClusterDistance(Feature const& fPx, Label lPx, Feature const& fCl, Label lCl) const;

    /**
     * Simple potts model.
     * @param l1 First label
     * @param l2 Second label
     * @return 0 in case the labels are identical, otherwise 1
     */
    template<typename T>
    inline float simplePotts(T l1, T l2) const;

private:
    UnaryFile m_unaryScores;
    Weights m_weights;
};

template<typename T>
float EnergyFunction::giveEnergy(LabelImage const& labeling, ColorImage<T> const& img, LabelImage const& sp) const
{
    auto unaryEnergy = giveUnaryEnergy(labeling);
    auto pairwiseEnergy = givePairwiseEnergy(labeling, img);
    auto spEnergy = giveSpEnergy(labeling, img, sp);
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
    float weight = std::exp(-m_weights.pairwiseSigmaSq() * colorDiffNormSq);
    return weight;
}

template<typename T>
float EnergyFunction::giveSpEnergy(LabelImage const& labeling, ColorImage<T> const& img, LabelImage const& sp) const
{
    size_t numSp = sp.minMax().second + 1;
    size_t numClasses = labeling.minMax().second + 1;

    // Find dominant labels and mean features
    std::vector<std::pair<Feature, size_t>> meanFeatures(numSp);
    std::vector<Label> dominantLabels(numSp);
    std::vector<std::vector<int>> labelFrequencies(numSp, std::vector<int>(numClasses, 0));
    for (size_t i = 0; i < sp.pixels(); ++i)
    {
        // Update label frequencies
        Label spLabel = sp.atSite(i);
        Label classLabel = labeling.atSite(i);
        labelFrequencies[spLabel][classLabel]++;

        // Update accumulated features
        Feature f(img, i);
        meanFeatures[spLabel].first += f;
        meanFeatures[spLabel].second++;
    }
    for (size_t i = 0; i < dominantLabels.size(); ++i)
    {
        // Computed dominant labels
        auto const& freq = labelFrequencies[i];
        dominantLabels[i] = std::distance(freq.begin(), std::max_element(freq.begin(), freq.end()));

        // Compute mean features
        meanFeatures[i].first /= meanFeatures[i].second;
    }

    float spEnergy = 0;
    for (size_t i = 0; i < labeling.pixels(); ++i)
        spEnergy += pixelToClusterDistance(Feature(img, i), labeling.atSite(i), meanFeatures[sp.atSite(i)].first,
                                           dominantLabels[sp.atSite(i)]);

    return spEnergy;
}

template<typename T>
float EnergyFunction::simplePotts(T l1, T l2) const
{
    if (l1 == l2)
        return 0;
    else
        return 1;
}

#endif //HSEG_ENERGYFUNCTION_H
