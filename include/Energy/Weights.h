//
// Created by jan on 23.08.16.
//

#ifndef HSEG_WEIGHTS_H
#define HSEG_WEIGHTS_H

#include <iostream>
#include <vector>
#include <Image/Image.h>
#include <typedefs.h>

using Weight = Cost;
using WeightVec = Eigen::VectorXf;

struct Stat
{
    float mean = 0;
    float stdev = 0;
    float max = 0;
    float min = 0;
    float mag = 0;
};

struct WeightStats
{
    Stat unary;
    Stat pairwise;
    Stat label;
    Stat feature;
    Stat total;
};

/**
 * Contains all (trainable) weights needed by the energy function
 */
class Weights
{
private:
    std::vector<WeightVec> m_unaryWeights;
    std::vector<WeightVec> m_pairwiseWeights;
    std::vector<WeightVec> m_higherOrderWeights;
    std::vector<WeightVec> m_featureWeights;

    friend class EnergyFunction;
    friend std::ostream& operator<<(std::ostream& stream, Weights const& weights);

public:
    /**
     * Default-constructs a weights vector (all zeros)
     * @param numLabels Amount of class labels
     * @param featDimPx Feature dimensionality of pixel features
     * @param featDimCluster Feature dimensionality of cluster features
     */
    explicit Weights(Label numClasses, uint32_t featDimPx, uint32_t featDimCluster);

    /**
     * Assigns random values between -1 and 1 to all weights
     */
    void randomize();

    /**
     * @return Amount of classes
     */
    inline size_t numClasses() const
    {
        return m_unaryWeights.size();
    }

    /**
     * Weight of the unary term
     * @param l Class label
     * @return The approriate weight
     */
    inline WeightVec const& unary(Label l) const
    {
        assert(l < m_unaryWeights.size());
        return m_unaryWeights[l];
    }

    /**
     * Weight of the pairwise term
     * @param l1 First label
     * @param l2 Second label
     * @return The approriate weight
     */
    inline WeightVec const& pairwise(Label l1, Label l2) const
    {
        size_t const index = l1 + l2 * numClasses();
        assert(index < m_pairwiseWeights.size());
        return m_pairwiseWeights[index];
    }

    /**
     * Weight of the higher order linear classifier
     * @param l1 First label
     * @param l2 Second label
     * @return The appropriate weight
     */
    inline WeightVec const& higherOrder(Label l1, Label l2) const
    {
        size_t const index = l1 + l2 * numClasses();
        assert(index < m_higherOrderWeights.size());
        return m_higherOrderWeights[index];
    }

    /**
     * Diagonal matrix of weights that act as covariance matrix
     * @param l1 Pixel label
     * @param l2 Cluster label
     * @return Feature similarity weight
     */
    inline WeightVec const& feature(Label l1, Label l2) const
    {
        size_t const index = l1 + l2 * numClasses();
        assert(index < m_featureWeights.size());
        return m_featureWeights[index];
    }

    inline WeightVec& unary(Label l)
    {
        assert(l < m_unaryWeights.size());
        return m_unaryWeights[l];
    }

    inline WeightVec& pairwise(Label l1, Label l2)
    {
        size_t const index = l1 + l2 * numClasses();
        assert(index < m_pairwiseWeights.size());
        return m_pairwiseWeights[index];
    }

    inline WeightVec& higherOrder(Label l1, Label l2)
    {
        size_t const index = l1 + l2 * numClasses();
        assert(index < m_higherOrderWeights.size());
        return m_higherOrderWeights[index];
    }

    inline WeightVec& feature(Label l1, Label l2)
    {
        size_t const index = l1 + l2 * numClasses();
        assert(index < m_featureWeights.size());
        return m_featureWeights[index];
    }

    /**
     * Sqares all elements;
     */
    void squareElements();

    /**
     * Compute element-wise sqrt in-place
     */
    void sqrt();

    /**
     * @return The squared norm of the vector
     */
    Weight sqNorm() const;

    /**
     * @return Weights for the regularizer
     */
    Weights regularized() const;

    /**
     * @return The sum of all vector elements
     */
    Weight sum() const;

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
     * Provides the mean values of the unary, pairwise, label consistency, feature similarity, and total weights
     * @return Mean weights
     */
    std::tuple<float, float, float, float, float> means() const;

    /**
     * Add element-wise
     * @param other Weight vector to add
     * @return Reference to this
     */
    Weights& operator+=(Weights const& other);

    /**
     * Add element-wise
     * @param other Weight vector to add
     * @return Result
     */
    Weights operator+(Weights const& other) const;

    /**
     * Add element-wise
     * @param bias Bias to add
     * @return Reference to this
     */
    Weights& operator+=(float bias);

    /**
     * Add element-wise
     * @param bias Bias to add
     * @return Result
     */
    Weights operator+(float bias) const;

    /**
     * Substract element-wise
     * @param other Weight vector to subtract
     * @return Reference to this
     */
    Weights& operator-=(Weights const& other);

    /**
     * Multiply element-wise
     * @param factor Factor
     * @return Reference to this
     */
    Weights& operator*=(float factor);

    /**
     * Dot-product
     * @param other Weights to multiply
     * @return Result
     */
    Weight operator*(Weights const& other) const;

    /**
     * @param factor Factor to multiply every weight with
     * @return Result
     */
    Weights operator*(float factor) const;

    /**
     * Divide element-wise
     * @param other Weight vector
     * @return Resulting weight vector
     */
    Weights& operator/=(Weights const& other);

    /**
     * Divide element-wise
     * @param other Weight vector
     * @return Resulting weight vector
     */
    Weights operator/(Weights const& other) const;

    /**
     * Compute statistics about these weights
     * @return The statistics
     */
    WeightStats computeStats() const;

    /**
     * Prints some statistics about the weight vector to a stream
     * @param out Output stream, defaults to std::cout
     */
    void printStats(std::ostream& out = std::cout) const;
};

std::ostream& operator<<(std::ostream& stream, Weights const& weights);



#endif //HSEG_WEIGHTS_H
