//
// Created by jan on 23.08.16.
//

#ifndef HSEG_WEIGHTS_H
#define HSEG_WEIGHTS_H


#include <vector>
#include <Image/Image.h>
#include <typedefs.h>

using Weight = Cost;
using WeightVec = Eigen::VectorXf;
using FeatSimMat = Eigen::MatrixXf;

/**
 * Contains all (trainable) weights needed by the energy function
 */
class Weights
{
private:
    std::vector<WeightVec> m_unaryWeights;
    std::vector<WeightVec> m_pairwiseWeights;
    std::vector<WeightVec> m_higherOrderWeights;
    FeatSimMat m_featureSimMat;
    FeatSimMat m_featureSimMatInv;

    friend class EnergyFunction;
    friend std::ostream& operator<<(std::ostream& stream, Weights const& weights);

    inline WeightVec& pairwise(Label l1, Label l2)
    {
        // Pairwise indices are stored as upper triangular matrix
        size_t const index = l1 + l2 * numClasses();
        assert(index < m_pairwiseWeights.size());
        return m_pairwiseWeights[index];
    }

    inline WeightVec& higherOrder(Label l1, Label l2)
    {
        // Pairwise indices are stored as upper triangular matrix
        size_t const index = l1 + l2 * numClasses();
        assert(index < m_higherOrderWeights.size());
        return m_higherOrderWeights[index];
    }

    void updateCachedInverseFeatMat();

public:
    /**
     * Default-constructs a weights vector (all zeros)
     * @param numLabels Amount of class labels
     * @param featDim Feature dimensionality
     */
    explicit Weights(Label numClasses, uint32_t featDim);

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
    inline WeightVec unary(Label l) const
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
        // Pairwise indices are stored as upper triangular matrix
        size_t const index = l1 + l2 * numClasses();
        assert(index < m_pairwiseWeights.size());
        return m_pairwiseWeights[index];
    }

    /**
     * Weight of the higher order linear classifier
     * @param l1 First label
     * @param l2 Second label
     * @return The approriate weight
     */
    inline WeightVec const& higherOrder(Label l1, Label l2) const
    {
        // Pairwise indices are stored as upper triangular matrix
        size_t const index = l1 + l2 * numClasses();
        assert(index < m_higherOrderWeights.size());
        return m_higherOrderWeights[index];
    }

    /**
     * @return Feature similarity matrix
     */
    inline FeatSimMat const& featureSimMat() const
    {
        return m_featureSimMat;
    }

    /**
     * @return Onverse feature similarity matrix
     */
    inline FeatSimMat const& featureSimMatInv() const
    {
        return m_featureSimMatInv;
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
};

std::ostream& operator<<(std::ostream& stream, Weights const& weights);



#endif //HSEG_WEIGHTS_H
