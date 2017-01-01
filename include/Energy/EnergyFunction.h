//
// Created by jan on 20.08.16.
//

#ifndef HSEG_ENERGYFUNCTION_H
#define HSEG_ENERGYFUNCTION_H


#include <Energy/UnaryFile.h>
#include <Inference/k-prototypes/Feature.h>
#include <Inference/k-prototypes/Cluster.h>
#include "WeightsVec.h"
#include "typedefs.h"

/**
 * Provides functionality to compute (partial) energies of the target energy function.
 * @note This will ignore pixels that have been assigned labels not in the range [0, classMax].
 */
class EnergyFunction
{
public:
    /**
     * Constructor
     * @param unaries Unary scores. The reference must stay valid as long as this object persists.
     * @param weights Weights to use. The reference must stay valid as long as this object persists.
     * @param pairwiseSigmaSq Sigma-Square inside of the exponential
     * @param featureWeights Matrix of feature weights
     */
    EnergyFunction(UnaryFile const& unaries, WeightsVec const& weights, Cost pairwiseSigmaSq, Matrix5 const& featureWeights);

    /**
     * Computes the overall energy
     * @param labeling Class labeling
     * @param img Color image
     * @param sp Superpixel labeling
     * @param clusters List of clusters
     * @return The energy of the given configuration
     */
    template<typename T>
    Cost giveEnergy(LabelImage const& labeling, ColorImage<T> const& img, LabelImage const& sp,
                     std::vector<Cluster> const& clusters) const;

    /**
     * Computes the overall energy by weights. I.e. to compute the actual energy, the result needs to be multiplied by
     * the weights vector.
     * @param labeling Class labeling
     * @param img Color image
     * @param sp Superpixel labeling
     * @param clusters List of clusters
     * @return The energy of the given configuration by weights
     */
    template<typename T>
    WeightsVec giveEnergyByWeight(LabelImage const& labeling, ColorImage<T> const& img, LabelImage const& sp,
                                  std::vector<Cluster> const& clusters) const;

    /**
    * Computes the unary energy by weights
    * @param labeling Class labeling
    * @param[out] energyW The unary energy of the given configuration will be stored here by (unary) weights
    * @return The unary energy of the given configuration
    */
    void computeUnaryEnergyByWeight(LabelImage const& labeling, WeightsVec& energyW) const;

    /**
     * Computes the pairwise energy by weights
     * @param labeling Class labeling
     * @param img Color image
     * @param[out] energyW The pairwise energy of the given configuration will be stored here by (pairwise) weights
     */
    template<typename T>
    void computePairwiseEnergyByWeight(LabelImage const& labeling, ColorImage<T> const& img, WeightsVec& energyW) const;

    /**
     * Computes the superpixel energy
     * @param labeling Class labeling
     * @param img Color image
     * @param sp Superpixel labeling
     * @param clusters List of clusters
     * @param[out] energyW The superpixel energy of the given configuration will be stored here by (superpixel) weights
     */
    template<typename T>
    void computeSpEnergyByWeight(LabelImage const& labeling, ColorImage<T> const& img, LabelImage const& sp,
                                 std::vector<Cluster> const& clusters, WeightsVec& energyW) const;

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
    inline Cost unaryCost(SiteId i, Label l) const
    {
        if(l >= m_unaryScores.classes())
            return 0;

        return m_weights.unary(l) * (-m_unaryScores.atSite(i, l));
    }

    /**
     * Computes the partial cost of a pairwise connection as given by the color of the pixels
     * @param img Color image
     * @param i First pixel id
     * @param j Second pixel id
     * @return The partial cost
     */
    template<typename T>
    Cost pairwisePixelWeight(ColorImage<T> const& img, SiteId i, SiteId j) const;

    /**
     * Computes the partial cost of a pairwise connection as given by the labels of the pixels
     * @param l1 First label
     * @param l2 Second label
     * @return The partial cost
     */
    inline Cost pairwiseClassWeight(Label l1, Label l2) const
    {
        if (l1 >= m_unaryScores.classes() || l2 >= m_unaryScores.classes())
            return 0;
        else
            return m_weights.pairwise(l1, l2);
    }

    /**
     * Computes the feature distance between two features
     * @param feature The first feature
     * @param feature2 The second feature
     * @return The feature distance
     */
    inline Cost featureDistance(Feature const& feature, Feature const& feature2) const
    {
        Vector5 f = feature.vec() - feature2.vec();
        Cost dist = f.transpose() * m_weights.feature() * f;
        return dist;
    }

    /**
     * Computes the feature distance by weight vector
     * @param feature The first feature
     * @param feature2 The second feature
     * @param weights Weights vector to write to
     */
    void featureDistanceByWeight(Feature const& feature, Feature const& feature2, WeightsVec& weights) const;

    /**
     * Computes the class label distance between two class labels
     * @param l1 First class label
     * @param l2 Second class label
     * @return The class label distance
     */
    inline Cost classDistance(Label l1, Label l2) const
    {
        if (l1 == l2 || l1 >= m_unaryScores.classes() || l2 >= m_unaryScores.classes())
            return 0;
        else
            return m_weights.classWeight(l1, l2);
    }

    /**
     * Additional superpixel data term
     * @param s Site on the image
     * @param l Label of the associated cluster
     * @return The additional data cost
     */
    inline Cost classData(SiteId /* s */, Label /* l */) const
    {
        return 0;
    }

    /**
     * Computes the pixel-to-cluster distance
     * @param fPx Feature of the pixel
     * @param lPx Class label of the pixel
     * @param cl Clusters
     * @param clusterId Cluster index
     * @return The pixel-to-cluster distance
     */
    inline Cost pixelToClusterDistance(Feature const& fPx, Label lPx, std::vector<Cluster> const& cl, Label clusterId) const
    {
        SiteId s = helper::coord::coordinateToSite(fPx.x(), fPx.y(), m_unaryScores.width());
        return featureDistance(fPx, cl[clusterId].mean) + classDistance(lPx, cl[clusterId].label) + classData(s, cl[clusterId].label);
    }

    /**
     * Simple potts model.
     * @param l1 First label
     * @param l2 Second label
     * @return 0 in case the labels are identical, otherwise 1
     */
    template<typename T>
    inline Cost simplePotts(T l1, T l2) const;

    /**
     * @return The unary file
     */
    inline UnaryFile const& unaryFile() const
    {
        return m_unaryScores;
    }

    /**
     * @return The weights
     */
    inline WeightsVec const& weights() const
    {
        return m_weights;
    }

protected:
    UnaryFile const& m_unaryScores;
    WeightsVec const& m_weights;
    Cost m_pairWiseSigmaSq;
    Matrix5 m_featureWeights;
};

template<typename T>
Cost EnergyFunction::giveEnergy(LabelImage const& labeling, ColorImage<T> const& img, LabelImage const& sp,
                                 std::vector<Cluster> const& clusters) const
{
    WeightsVec energy = m_weights;
    energy *= giveEnergyByWeight(labeling, img, sp, clusters);
    return energy.sum();
}

template<typename T>
void EnergyFunction::computePairwiseEnergyByWeight(LabelImage const& labeling, ColorImage<T> const& img,
                                                   WeightsVec& energyW) const
{
    for (Coord x = 0; x < labeling.width(); ++x)
    {
        for (Coord y = 0; y < labeling.height(); ++y)
        {
            Label l = labeling.at(x, y);
            if(x + 1 < labeling.width())
            {
                Label lR = labeling.at(x + 1, y);
                if (l != lR && l < m_unaryScores.classes() && lR < m_unaryScores.classes())
                {
                    auto energy = pairwisePixelWeight(img, helper::coord::coordinateToSite(x, y, labeling.width()),
                                                      helper::coord::coordinateToSite(x + 1, y, labeling.width()));
                    energyW.pairwise(l, lR) += energy;
                }
            }

            if(y + 1 < labeling.height())
            {
                Label lD = labeling.at(x, y + 1);
                if (l != lD && l < m_unaryScores.classes() && lD < m_unaryScores.classes())
                {
                    auto energy = pairwisePixelWeight(img, helper::coord::coordinateToSite(x, y, labeling.width()),
                                                      helper::coord::coordinateToSite(x, y + 1, labeling.width()));
                    energyW.pairwise(l, lD) += energy;
                }
            }
        }
    }
}

template<typename T>
Cost EnergyFunction::pairwisePixelWeight(ColorImage<T> const& img, SiteId i, SiteId j) const
{
    Cost rDiff = img.atSite(i, 0) - img.atSite(j, 0);
    Cost gDiff = img.atSite(i, 1) - img.atSite(j, 1);
    Cost bDiff = img.atSite(i, 2) - img.atSite(j, 2);
    Cost colorDiffNormSq = rDiff * rDiff + gDiff * gDiff + bDiff * bDiff;
    Cost weight = std::exp(-m_pairWiseSigmaSq * colorDiffNormSq);
    return weight;
}

template<typename T>
void EnergyFunction::computeSpEnergyByWeight(LabelImage const& labeling, ColorImage<T> const& img, LabelImage const& sp,
                                             std::vector<Cluster> const& clusters, WeightsVec& energyW) const
{
    for (SiteId i = 0; i < labeling.pixels(); ++i)
    {
        assert(clusters.size() > sp.atSite(i));

        // Only consider pixels with a valid label
        if (labeling.atSite(i) < m_unaryScores.classes())
        {
            auto const& cluster = clusters[sp.atSite(i)];
            if (labeling.atSite(i) != cluster.label)
                energyW.classWeight(labeling.atSite(i), clusters[sp.atSite(i)].label)++;
            featureDistanceByWeight(Feature(img, i), cluster.mean, energyW);
        }
    }
}

template<typename T>
WeightsVec EnergyFunction::giveEnergyByWeight(LabelImage const& labeling, ColorImage<T> const& img,
                                              LabelImage const& sp,
                                              std::vector<Cluster> const& clusters) const
{
    WeightsVec w(m_unaryScores.classes(), false); // Zero-initialized weights

    computeUnaryEnergyByWeight(labeling, w);
    computePairwiseEnergyByWeight(labeling, img, w);
    computeSpEnergyByWeight(labeling, img, sp, clusters, w);

    return w;
}

template<typename T>
inline Cost EnergyFunction::simplePotts(T l1, T l2) const
{
    if (l1 == l2)
        return 0;
    else
        return 1;
}

#endif //HSEG_ENERGYFUNCTION_H
