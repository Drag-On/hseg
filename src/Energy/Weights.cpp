//
// Created by jan on 23.08.16.
//

#include <fstream>
#include <iomanip>
#include <cmath>
#include <iostream>
#include "Energy/Weights.h"

Weights::Weights(Label numClasses, uint32_t featDim)
{
    m_unaryWeights.resize(numClasses, WeightVec::Zero(featDim + 1)); // +1 for the bias
    m_pairwiseWeights.resize(numClasses * numClasses, WeightVec::Zero(2 * featDim + 1));
    m_higherOrderWeights.resize(numClasses * numClasses, WeightVec::Zero(2 * featDim + 1));
    m_featureWeights.resize(numClasses * numClasses, WeightVec::Zero(featDim));

    clampToFeasible();
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
    for (size_t i = 0; i < m_featureWeights.size(); ++i)
        m_featureWeights[i] += other.m_featureWeights[i];

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
    for (size_t i = 0; i < m_featureWeights.size(); ++i)
        m_featureWeights[i] = m_featureWeights[i].array() + bias;

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
    for (size_t i = 0; i < m_featureWeights.size(); ++i)
        m_featureWeights[i] -= other.m_featureWeights[i];

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
    for (size_t i = 0; i < m_featureWeights.size(); ++i)
        m_featureWeights[i] *= factor;

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
    for (size_t i = 0; i < m_featureWeights.size(); ++i)
        result += m_featureWeights[i].dot(other.m_featureWeights[i]);

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
    for (size_t i = 0; i < m_featureWeights.size(); ++i)
        m_featureWeights[i].cwiseQuotient(other.m_featureWeights[i]);

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
    for (size_t i = 0; i < m_featureWeights.size(); ++i)
        m_featureWeights[i] = m_featureWeights[i].cwiseProduct(m_featureWeights[i]);
}

void Weights::sqrt()
{
    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] = m_unaryWeights[i].cwiseSqrt();
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] =  m_pairwiseWeights[i].cwiseSqrt();
    for (size_t i = 0; i < m_higherOrderWeights.size(); ++i)
        m_higherOrderWeights[i] = m_higherOrderWeights[i].cwiseSqrt();
    for (size_t i = 0; i < m_featureWeights.size(); ++i)
        m_featureWeights[i] = m_featureWeights[i].cwiseSqrt();
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
    for (size_t i = 0; i < m_featureWeights.size(); ++i)
        sqNorm += m_featureWeights[i].squaredNorm();

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
    for (size_t i = 0; i < m_featureWeights.size(); ++i)
        sum += m_featureWeights[i].sum();

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
    for (size_t i = 0; i < weights.numClasses(); ++i)
    {
        for (size_t j = 0; j < weights.numClasses(); ++j)
        {
            stream << std::setw(2) << i << "," << std::setw(2) << j << ": ";
            stream << std::setw(6) << weights.feature(i, j).transpose();
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
        out.write("WEIGHT03", 8);
        uint32_t featDim = m_unaryWeights[0].size() - 1;
        uint32_t noUnaries = m_unaryWeights.size();
        uint32_t noPairwise = m_pairwiseWeights.size();
        uint32_t noHigherOrder = m_higherOrderWeights.size();
        uint32_t noFeature = m_featureWeights.size();
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
        for(auto const& e : m_featureWeights)
        {
            assert(e.size() == featDim);
            out.write(reinterpret_cast<const char*>(e.data()), sizeof(e(0)) * e.size());
        }
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
        if(std::strncmp(id, "WEIGHT03", 8) != 0)
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
        m_unaryWeights.resize(noUnaries, WeightVec::Zero(featDim + 1));
        m_pairwiseWeights.resize(noPairwise, WeightVec::Zero(featDim * 2 + 1));
        m_higherOrderWeights.resize(noHigherOrder, WeightVec::Zero(featDim * 2 + 1));
        m_featureWeights.resize(noFeature, WeightVec::Zero(featDim));
        for(auto& e : m_unaryWeights)
            in.read(reinterpret_cast<char*>(e.data()), sizeof(e(0)) * (featDim + 1));
        for(auto& e : m_pairwiseWeights)
            in.read(reinterpret_cast<char*>(e.data()), sizeof(e(0)) * (featDim * 2 + 1));
        for(auto& e : m_higherOrderWeights)
            in.read(reinterpret_cast<char*>(e.data()), sizeof(e(0)) * (featDim * 2 + 1));
        for(auto& e : m_featureWeights)
            in.read(reinterpret_cast<char*>(e.data()), sizeof(e(0)) * (featDim));
        in.close();

        return true;
    }
    return false;
}

std::tuple<float, float, float, float, float> Weights::means() const
{
    float meanUnary = 0, meanPairwise = 0, meanLabelCons = 0, meanFeature = 0, meanTotal = 0;

    for(size_t i = 0; i < m_unaryWeights.size(); ++i)
        meanUnary += m_unaryWeights[i].mean();

    for(size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        meanPairwise += m_pairwiseWeights[i].mean();

    for(size_t i = 0; i < m_higherOrderWeights.size(); ++i)
        meanLabelCons += m_higherOrderWeights[i].mean();

    for(size_t i = 0; i < m_featureWeights.size(); ++i)
        meanFeature += m_featureWeights[i].mean();

    meanTotal = meanUnary + meanPairwise + meanLabelCons + meanFeature;

    meanUnary /= m_unaryWeights.size();
    meanPairwise /= m_pairwiseWeights.size();
    meanLabelCons /= m_higherOrderWeights.size();
    meanFeature /= m_featureWeights.size();
    meanTotal /= m_unaryWeights.size() + m_pairwiseWeights.size() + m_higherOrderWeights.size() + m_featureWeights.size();

    return std::tie(meanUnary, meanPairwise, meanLabelCons, meanFeature, meanTotal);
}

void Weights::clampToFeasible()
{
    static const float threshold = 1e-6;
    for(size_t i = 0; i < m_featureWeights.size(); ++i)
    {
        WeightVec& w = m_featureWeights[i];
        for(size_t j = 0; j < w.size(); ++j)
        {
            if(w(j) >= 0 && w(j) < threshold)
                w(j) = threshold;
            else if(w(j) < 0 && w(j) >-threshold)
                w(j) = -threshold;
        }
    }

}

void Weights::randomize()
{
    for(size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] = WeightVec::Random(m_unaryWeights[i].size());
    for(size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] = WeightVec::Random(m_pairwiseWeights[i].size());
    for(size_t i = 0; i < m_higherOrderWeights.size(); ++i)
        m_higherOrderWeights[i] = WeightVec::Random(m_higherOrderWeights[i].size());
    for(size_t i = 0; i < m_featureWeights.size(); ++i)
        m_featureWeights[i] = WeightVec::Random(m_featureWeights[i].size());
}

void Weights::printStats(std::ostream& out) const
{
    auto printer = [&out] (std::string const& type, std::vector<WeightVec> const& w)
    {
        float mean = w[0].mean();
        float max = w[0].maxCoeff();
        float min = w[0].minCoeff();
        float mag = w[0].norm();

        for(size_t i = 1; i < w.size(); ++i)
        {
            mean += w[i].mean();
            float m = w[i].maxCoeff();
            if(m > max)
                max = m;
            m = w[i].minCoeff();
            if(m < min)
                min = m;
            mag += w[i].norm();
        }
        mean /= w.size();
        mag /= w.size();
        out << type << std::endl;
        out << " mean = " << mean << std::endl;
        out << " max = " << max << std::endl;
        out << " min = " << min << std::endl;
        out << " mag = " << mag << std::endl;
    };

    printer("UNARY", m_unaryWeights);
    printer("PAIRWISE", m_pairwiseWeights);
    printer("LABELCON", m_higherOrderWeights);
    printer("FEATURE", m_featureWeights);
}
