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

/**
 * Contains all (trainable) weights needed by the energy function
 */
class Weights
{
private:
    std::vector<WeightVec> m_unaryWeights;
    std::vector<WeightVec> m_pairwiseWeights;

    friend class EnergyFunction;
    friend std::ostream& operator<<(std::ostream& stream, Weights const& weights);

    inline WeightVec& pairwise(Label l1, Label l2)
    {
        // Pairwise indices are stored as upper triangular matrix
        size_t const index = l1 + l2 * numClasses();
        assert(index < m_pairwiseWeights.size());
        return m_pairwiseWeights[index];
    }

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
};

std::ostream& operator<<(std::ostream& stream, Weights const& weights);



#endif //HSEG_WEIGHTS_H
