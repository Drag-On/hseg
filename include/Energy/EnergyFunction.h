//
// Created by jan on 20.08.16.
//

#ifndef HSEG_ENERGYFUNCTION_H
#define HSEG_ENERGYFUNCTION_H


#include <Energy/UnaryFile.h>
#include <Inference/k-prototypes/Feature.h>
#include <Inference/k-prototypes/Cluster.h>
#include "WeightsVec.h"
#include <Eigen/Dense>

using Matrix5f = Eigen::Matrix<float, 5, 5>;
using Vector5f = Eigen::Matrix<float, 5, 1>;

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
    EnergyFunction(UnaryFile const& unaries, WeightsVec const& weights, float pairwiseSigmaSq, Matrix5f const& featureWeights);

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
     * Computes the overall energy by weights. I.e. to compute the actual energy, the result needs to be muliplied by
     * the weights vector.
     *  @param labeling Class labeling
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
    float featureDistance(Feature const& feature, Feature const& feature2) const;

    /**
     * Computes the class label distance between two class labels
     * @param l1 First class label
     * @param l2 Second class label
     * @return The class label distance
     */
    inline float classDistance(Label l1, Label l2) const
    {
        if (l1 == l2 || l1 >= m_unaryScores.classes() || l2 >= m_unaryScores.classes())
            return 0;
        else
            return m_weights.classWeight(l1, l2);
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

    /**
     * @return The weights
     */
    WeightsVec const& weights() const;

protected:
    UnaryFile const& m_unaryScores;
    WeightsVec const& m_weights;
    float m_pairWiseSigmaSq;
    Matrix5f m_featureWeights;
};

template<typename T>
float EnergyFunction::giveEnergy(LabelImage const& labeling, ColorImage<T> const& img, LabelImage const& sp,
                                 std::vector<Cluster> const& clusters) const
{
    WeightsVec energy = m_weights;
    energy *= giveEnergyByWeight(labeling, img, sp, clusters);

    // The feature distance is missing from the above since there is no weight for it
    // Thus, add it here
    float featureEnergy = 0;
    for(size_t i = 0; i < labeling.pixels(); ++i)
    {
        Feature f(img, i);
        Label l = sp.atSite(i);
        featureEnergy += featureDistance(f, clusters[l].mean);
    }

    return energy.sum() + featureEnergy;
}

template<typename T>
void EnergyFunction::computePairwiseEnergyByWeight(LabelImage const& labeling, ColorImage<T> const& img,
                                                   WeightsVec& energyW) const
{
    for (size_t x = 0; x < labeling.width(); ++x)
    {
        for (size_t y = 0; y < labeling.height(); ++y)
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
void EnergyFunction::computeSpEnergyByWeight(LabelImage const& labeling, ColorImage<T> const& /* img */, LabelImage const& sp,
                                             std::vector<Cluster> const& clusters, WeightsVec& energyW) const
{
    for (size_t i = 0; i < labeling.pixels(); ++i)
    {
        assert(clusters.size() > sp.atSite(i));

        // Only consider pixels with a valid label
        if (labeling.atSite(i) < m_unaryScores.classes())
        {
            if (labeling.atSite(i) != clusters[sp.atSite(i)].label)
                energyW.classWeight(labeling.atSite(i), clusters[sp.atSite(i)].label)++;
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

    w *= m_weights;

    return w;
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
