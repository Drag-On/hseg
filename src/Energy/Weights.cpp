//
// Created by jan on 23.08.16.
//

#include <fstream>
#include <iomanip>
#include <cmath>
#include "Energy/Weights.h"

Weights::Weights(Label numClasses, uint32_t featDim)
{
    m_unaryWeights.resize(numClasses, WeightVec::Zero(featDim + 1)); // +1 for the bias
    m_pairwiseWeights.resize(numClasses * numClasses, WeightVec::Zero(2 * featDim + 1));
    m_higherOrderWeights.resize(numClasses * numClasses, WeightVec::Zero(2 * featDim + 1));
    m_featureSimMat = m_featureSimMatInv = FeatSimMat::Identity(featDim, featDim);
}

Weights& Weights::operator+=(Weights const& other)
{
    assert(numClasses() == other.numClasses());

    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] += other.m_unaryWeights[i];
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] += other.m_pairwiseWeights[i];
    for (size_t i = 0; i < m_higherOrderWeights.size(); ++i)
        m_higherOrderWeights[i] += other.m_higherOrderWeights[i];
    m_featureSimMat += other.m_featureSimMat;

    return *this;
}

Weights Weights::operator+(Weights const& other) const
{
    Weights result = *this;
    result += other;
    return result;
}

Weights& Weights::operator+=(float bias)
{
    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] = m_unaryWeights[i].array() + bias;
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] = m_pairwiseWeights[i].array() + bias;
    for (size_t i = 0; i < m_higherOrderWeights.size(); ++i)
        m_higherOrderWeights[i] = m_higherOrderWeights[i].array() + bias;
    m_featureSimMat = m_featureSimMat.array() + bias;

    return *this;
}

Weights Weights::operator+(float bias) const
{
    Weights result = *this;
    result += bias;
    return result;
}

Weights& Weights::operator-=(Weights const& other)
{
    assert(numClasses() == other.numClasses());

    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] -= other.m_unaryWeights[i];
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] -= other.m_pairwiseWeights[i];
    for (size_t i = 0; i < m_higherOrderWeights.size(); ++i)
        m_higherOrderWeights[i] -= other.m_higherOrderWeights[i];
    m_featureSimMat -= other.m_featureSimMat;

    return *this;
}

Weights& Weights::operator*=(float factor)
{
    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] *= factor;
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] *= factor;
    for (size_t i = 0; i < m_higherOrderWeights.size(); ++i)
        m_higherOrderWeights[i] *= factor;
    m_featureSimMat *= factor;

    return *this;
}

Weight Weights::operator*(Weights const& other) const
{
    assert(numClasses() == other.numClasses());

    Weight result = 0;

    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        result += m_unaryWeights[i].dot(other.m_unaryWeights[i]);
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        result += m_pairwiseWeights[i].dot(other.m_pairwiseWeights[i]);
    for (size_t i = 0; i < m_higherOrderWeights.size(); ++i)
        result += m_higherOrderWeights[i].dot(other.m_higherOrderWeights[i]);
    result += m_featureSimMat.cwiseProduct(other.m_featureSimMat).sum();

    return result;
}

Weights Weights::operator*(float factor) const
{
    Weights result = *this;
    result *= factor;
    return result;
}

Weights& Weights::operator/=(Weights const& other)
{
    assert(numClasses() == other.numClasses());

    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i].cwiseQuotient(other.m_unaryWeights[i]);
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i].cwiseQuotient(other.m_pairwiseWeights[i]);
    for (size_t i = 0; i < m_higherOrderWeights.size(); ++i)
        m_higherOrderWeights[i].cwiseQuotient(other.m_higherOrderWeights[i]);
    m_featureSimMat.cwiseQuotient(other.m_featureSimMat);

    return *this;
}

Weights Weights::operator/(Weights const& other) const
{
    Weights result = *this;
    result /= other;
    return result;
}

void Weights::squareElements()
{
    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] = m_unaryWeights[i].cwiseProduct(m_unaryWeights[i]);
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] =  m_pairwiseWeights[i].cwiseProduct(m_pairwiseWeights[i]);
    for (size_t i = 0; i < m_higherOrderWeights.size(); ++i)
        m_higherOrderWeights[i] = m_higherOrderWeights[i].cwiseProduct(m_higherOrderWeights[i]);
    m_featureSimMat = m_featureSimMat.cwiseProduct(m_featureSimMat);
}

void Weights::sqrt()
{
    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] = m_unaryWeights[i].cwiseSqrt();
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] =  m_pairwiseWeights[i].cwiseSqrt();
    for (size_t i = 0; i < m_higherOrderWeights.size(); ++i)
        m_higherOrderWeights[i] = m_higherOrderWeights[i].cwiseSqrt();
    m_featureSimMat = m_featureSimMat.cwiseSqrt();
}

Weight Weights::sqNorm() const
{
    Weight sqNorm = 0;

    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        sqNorm += m_unaryWeights[i].squaredNorm();
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        sqNorm += m_pairwiseWeights[i].squaredNorm();
    for (size_t i = 0; i < m_higherOrderWeights.size(); ++i)
        sqNorm += m_higherOrderWeights[i].squaredNorm();
    sqNorm += m_featureSimMat.squaredNorm();

    return sqNorm;
}

Weight Weights::sum() const
{
    Weight sum = 0;

    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        sum += m_unaryWeights[i].sum();
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        sum += m_pairwiseWeights[i].sum();
    for (size_t i = 0; i < m_higherOrderWeights.size(); ++i)
        sum += m_higherOrderWeights[i].sum();
    sum += m_featureSimMat.sum();

    return sum;
}

std::ostream& operator<<(std::ostream& stream, Weights const& weights)
{
    stream.precision(4);
    stream << std::fixed;

    stream << "unary:" << std::endl;
    for (size_t i = 0; i < weights.numClasses(); ++i)
    {
        stream << std::setw(2) << i << ": ";
        stream << std::setw(6) << weights.m_unaryWeights[i].transpose();
        stream << std::endl;
    }
    stream << std::endl << std::endl;

    stream << "pairwise:" << std::endl;
    for (size_t i = 0; i < weights.numClasses(); ++i)
    {
        for (size_t j = 0; j < weights.numClasses(); ++j)
        {
            stream << std::setw(2) << i << "," << std::setw(2) << j << ": ";
            stream << std::setw(6) << weights.pairwise(i, j).transpose();
            stream << std::endl;
        }
    }
    stream << std::endl << std::endl;

    stream << "higher-order:" << std::endl;
    for (size_t i = 0; i < weights.numClasses(); ++i)
    {
        for (size_t j = 0; j < weights.numClasses(); ++j)
        {
            stream << std::setw(2) << i << "," << std::setw(2) << j << ": ";
            stream << std::setw(6) << weights.higherOrder(i, j).transpose();
            stream << std::endl;
        }
    }
    stream << std::endl << std::endl;

    stream << "feature:" << std::endl;
    stream << weights.m_featureSimMat << std::endl;
    stream << std::endl << std::endl;

    return stream;
}

bool Weights::write(std::string const& filename) const
{
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    if(out.is_open())
    {
        out.write("WEIGHT01", 8);
        uint32_t featDim = m_unaryWeights[0].size() - 1;
        uint32_t noUnaries = m_unaryWeights.size();
        uint32_t noPairwise = m_pairwiseWeights.size();
        uint32_t noHigherOrder = m_higherOrderWeights.size();
        uint32_t noFeature = 1;
        out.write(reinterpret_cast<const char*>(&featDim), sizeof(featDim));
        out.write(reinterpret_cast<const char*>(&noUnaries), sizeof(noUnaries));
        out.write(reinterpret_cast<const char*>(&noPairwise), sizeof(noPairwise));
        out.write(reinterpret_cast<const char*>(&noHigherOrder), sizeof(noHigherOrder));
        out.write(reinterpret_cast<const char*>(&noFeature), sizeof(noFeature));
        for(auto const& e : m_unaryWeights)
        {
            assert(e.size() == featDim + 1);
            out.write(reinterpret_cast<const char*>(e.data()), sizeof(e(0)) * e.size());
        }
        for(auto const& e : m_pairwiseWeights)
        {
            assert(e.size() == featDim * 2 + 1);
            out.write(reinterpret_cast<const char*>(e.data()), sizeof(e(0)) * e.size());
        }
        for(auto const& e : m_higherOrderWeights)
        {
            assert(e.size() == featDim * 2 + 1);
            out.write(reinterpret_cast<const char*>(e.data()), sizeof(e(0)) * e.size());
        }
        out.write(reinterpret_cast<const char*>(m_featureSimMat.data()), sizeof(m_featureSimMat(0,0)) * m_featureSimMat.size());
        out.close();
        return true;
    }
    return false;
}

bool Weights::read(std::string const& filename)
{
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if(in.is_open())
    {
        char id[8];
        in.read(id, 8);
        if(std::strncmp(id, "WEIGHT01", 8) != 0)
        {
            in.close();
            return false;
        }
        uint32_t featDim, noUnaries, noPairwise, noHigherOrder, noFeature;
        in.read(reinterpret_cast<char*>(&featDim), sizeof(featDim));
        in.read(reinterpret_cast<char*>(&noUnaries), sizeof(noUnaries));
        in.read(reinterpret_cast<char*>(&noPairwise), sizeof(noPairwise));
        in.read(reinterpret_cast<char*>(&noHigherOrder), sizeof(noHigherOrder));
        in.read(reinterpret_cast<char*>(&noFeature), sizeof(noFeature));
        assert(noFeature == 1);
        m_unaryWeights.resize(noUnaries, WeightVec::Zero(featDim + 1));
        m_pairwiseWeights.resize(noPairwise, WeightVec::Zero(featDim * 2 + 1));
        m_higherOrderWeights.resize(noPairwise, WeightVec::Zero(featDim * 2 + 1));
        for(auto& e : m_unaryWeights)
            in.read(reinterpret_cast<char*>(e.data()), sizeof(e(0)) * (featDim + 1));
        for(auto& e : m_pairwiseWeights)
            in.read(reinterpret_cast<char*>(e.data()), sizeof(e(0)) * (featDim * 2 + 1));
        for(auto& e : m_higherOrderWeights)
            in.read(reinterpret_cast<char*>(e.data()), sizeof(e(0)) * (featDim * 2 + 1));
        in.read(reinterpret_cast<char*>(m_featureSimMat.data()), sizeof(m_featureSimMat(0, 0)) * (featDim * featDim));
        in.close();

        updateCachedInverseFeatMat();
        return true;
    }
    return false;
}

void Weights::clampToFeasible()
{
    // Feature weights must be positive definite
    Eigen::SelfAdjointEigenSolver<FeatSimMat> es(m_featureSimMat);
    FeatSimMat D = es.eigenvalues().cast<float>().asDiagonal();
    FeatSimMat V = es.eigenvectors().cast<float>();
    for(uint16_t i = 0; i < es.eigenvalues().size(); ++i)
        if(D(i, i) < 1e-5f)
            D(i, i) = 1e-5f;
    m_featureSimMat = V * D * V.inverse();

    updateCachedInverseFeatMat();
}

void Weights::randomize()
{
    for(size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] = WeightVec::Random(m_unaryWeights[i].size());
    for(size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] = WeightVec::Random(m_pairwiseWeights[i].size());
    for(size_t i = 0; i < m_higherOrderWeights.size(); ++i)
        m_higherOrderWeights[i] = WeightVec::Random(m_higherOrderWeights[i].size());
    m_featureSimMat = FeatSimMat::Random(m_featureSimMat.rows(), m_featureSimMat.cols());
}

void Weights::updateCachedInverseFeatMat()
{
    // Compute inverse of the feature similarity matrix
    m_featureSimMatInv = m_featureSimMat.inverse();
}
