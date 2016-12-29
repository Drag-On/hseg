//
// Created by jan on 23.08.16.
//

#ifndef HSEG_WEIGHTS_H
#define HSEG_WEIGHTS_H


#include <vector>
#include <Image/Image.h>
#include <typedefs.h>
#include <Eigen/Dense>

using Weight = Cost;
using Matrix5 = Eigen::Matrix<Cost, 5, 5>;

/**
 * Contains all (trainable) weights needed by the energy function
 */
class WeightsVec
{
private:
    Label m_numLabels;
    uint16_t m_numFeatures = 5;
    std::vector<float> m_unaryWeights;
    std::vector<float> m_pairwiseWeights;
    std::vector<float> m_featureWeights;
    std::vector<float> m_classWeights;

    friend class EnergyFunction;
    friend std::ostream& operator<<(std::ostream& stream, WeightsVec const& weights);

    float& pairwise(Label l1, Label l2);

    float& classWeight(Label l1, Label l2);

public:
    /**
     * Creates default weights
     * @param numLabels Amount of class labels
     * @param defaultInit Determines if the weights will be initialized with default values or with zeros
     */
    explicit WeightsVec(Label numLabels, bool defaultInit = true);

    /**
     * Initialize some basic weights
     * @param numLabels Amount of class labels
     * @param unary Weight of the unary term
     * @param pairwise Weight of the pairwise term
     * @param feature Weight of the feature term
     * @param label Label weight
     */
    WeightsVec(Label numLabels, Weight unary, Weight pairwise, Weight feature, Weight label);

    /**
     * Weight of the unary term
     * @param l Class label
     * @return The approriate weight
     */
    inline Weight unary(Label l) const
    {
        assert(l < m_unaryWeights.size());
        return m_unaryWeights[l];
    }

    /**
     * @return True in case any pairwise weight is non-zero (when rounded), otherwise false
     */
    bool hasPairwiseWeight() const;

    /**
     * Weight of the pairwise term
     * @param l1 First label
     * @param l2 Second label
     * @return The approriate weight
     */
    Weight pairwise(Label l1, Label l2) const;

    /**
     * Weight of the feature term
     * @return Feature weight
     */
    inline Matrix5 feature() const
    {
        Matrix5 featureWeights;
        for (uint16_t i = 0; i < m_numFeatures; ++i)
        {
            for(uint16_t j = i; j < m_numFeatures; ++j)
            {
                assert(i + (j + 1) * j / 2 < m_featureWeights.size());
                featureWeights(i, j) = featureWeights(j, i) = m_featureWeights[i + (j + 1) * j / 2];
            }
        }
        return featureWeights;
    }

    /**
     * Grant access to a single feature weight
     * @param i First index
     * @param j Second index
     * @return Reference to the appropriate entry
     */
    inline float& featureWeight(uint16_t i, uint16_t j)
    {
        assert(i < m_featureWeights.size() && j < m_featureWeights.size());
        if(i > j)
            std::swap(i, j);
        return m_featureWeights[i + (j + 1) * j / 2];
    }

    /**
     * @return The class weight
     */
    Weight classWeight(Label l1, Label l2) const;

    /**
     * @return Sum of all coefficients
     */
    inline Weight sum() const
    {
        return sumUnary() + sumPairwise() + sumSuperpixel();
    }

    /**
     * @return Sum of all unary coefficients
     */
    Weight sumUnary() const;

    /**
     * @return Sum of all pairwise coefficients
     */
    Weight sumPairwise() const;

    /**
     * @return Sum of all superpixel coefficients
     */
    Weight sumSuperpixel() const;

    /**
     * @return The squared norm of the vector
     */
    Weight sqNorm() const;

    /**
     * Clamps all values to the closest feasible value
     */
    void clampToFeasible();

    /**
     * Writes the weights vector to a file on harddisk
     * @param filename File to write to
     * @return True in case the file has been written properly, otherwise false
     */
    bool write(std::string const& filename) const;

    /**
     * Reads the weights vector from a file on harddisk
     * @param filename File to read from
     * @return True in case the file has been read properly, otherwise false
     */
    bool read(std::string const& filename);

    /**
     * Add element-wise
     * @param other Weight vector to add
     * @return Reference to this
     */
    WeightsVec& operator+=(WeightsVec const& other);

    /**
     * Substract element-wise
     * @param other Weight vector to subtract
     * @return Reference to this
     */
    WeightsVec& operator-=(WeightsVec const& other);

    /**
     * Multiply element-wise
     * @param factor Factor
     * @return Reference to this
     */
    WeightsVec& operator*=(float factor);

    /**
     * Multiply element-wise
     * @param other Weights to multiply
     * @return Reference to this
     */
    WeightsVec& operator*=(WeightsVec const& other);

    std::vector<float>& unaryWeights();

    std::vector<float>& pairwiseWeights();

    std::vector<float>& classWeights();
};

std::ostream& operator<<(std::ostream& stream, WeightsVec const& weights);



#endif //HSEG_WEIGHTS_H
