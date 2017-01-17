//
// Created by jan on 23.08.16.
//

#include <fstream>
#include <iomanip>
#include "Energy/Weights.h"

Weights::Weights(Label numClasses, uint32_t featDim)
{
    m_unaryWeights.resize(numClasses, WeightVec::Zero(featDim + 1)); // +1 for the bias
    m_pairwiseWeights.resize(numClasses * numClasses, WeightVec::Zero(2 * featDim + 1));
}

Weights& Weights::operator+=(Weights const& other)
{
    assert(numClasses() == other.numClasses());

    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] += other.m_unaryWeights[i];
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] += other.m_pairwiseWeights[i];

    return *this;
}

Weights& Weights::operator-=(Weights const& other)
{
    assert(numClasses() == other.numClasses());

    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] -= other.m_unaryWeights[i];
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] -= other.m_pairwiseWeights[i];

    return *this;
}

Weights& Weights::operator*=(float factor)
{
    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] *= factor;
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] *= factor;

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

    return result;
}

Weight Weights::sqNorm() const
{
    Weight sqNorm = 0;

    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        sqNorm += m_unaryWeights[i].squaredNorm();
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        sqNorm += m_pairwiseWeights[i].squaredNorm();

    return sqNorm;
}

Weight Weights::sum() const
{
    Weight sum = 0;

    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        sum += m_unaryWeights[i].sum();
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        sum += m_pairwiseWeights[i].sum();

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

    return stream;
}

bool Weights::write(std::string const& filename) const
{
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    if(out.is_open())
    {
        out.write("WEIGHT00", 8);
        uint32_t featDim = m_unaryWeights[0].size();
        uint32_t noUnaries = m_unaryWeights.size();
        uint32_t noPairwise = m_pairwiseWeights.size();
        out.write(reinterpret_cast<const char*>(&featDim), sizeof(featDim));
        out.write(reinterpret_cast<const char*>(&noUnaries), sizeof(noUnaries));
        out.write(reinterpret_cast<const char*>(&noPairwise), sizeof(noPairwise));
        for(auto const& e : m_unaryWeights)
            out.write(reinterpret_cast<const char*>(e.data()), sizeof(e(0)) * featDim);
        for(auto const& e : m_pairwiseWeights)
            out.write(reinterpret_cast<const char*>(e.data()), sizeof(e(0)) * featDim);
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
        if(std::strncmp(id, "WEIGHT00", 8) != 0)
        {
            in.close();
            return false;
        }
        uint32_t featDim;
        uint32_t noUnaries;
        uint32_t noPairwise;
        in.read(reinterpret_cast<char*>(&featDim), sizeof(featDim));
        in.read(reinterpret_cast<char*>(&noUnaries), sizeof(noUnaries));
        in.read(reinterpret_cast<char*>(&noPairwise), sizeof(noPairwise));
        m_unaryWeights.resize(noUnaries, WeightVec::Zero(featDim));
        m_pairwiseWeights.resize(noPairwise, WeightVec::Zero(featDim));
        for(auto& e : m_unaryWeights)
            in.read(reinterpret_cast<char*>(e.data()), sizeof(e(0)) * featDim);
        for(auto& e : m_pairwiseWeights)
            in.read(reinterpret_cast<char*>(e.data()), sizeof(e(0)) * featDim);
        in.close();
        return true;
    }
    return false;
}

void Weights::clampToFeasible()
{
    // For now, all weights are permitted
}

void Weights::randomize()
{
    for(size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] = WeightVec::Random(m_unaryWeights[i].size());
    for(size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] = WeightVec::Random(m_pairwiseWeights[i].size());
}
