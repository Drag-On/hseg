//
// Created by jan on 23.08.16.
//

#include <fstream>
#include <iomanip>
#include <cmath>
#include <iostream>
#include "Energy/Weights.h"

Weights::Weights(Label numClasses, uint32_t featDimPx, uint32_t featDimCluster)
{
    m_unaryWeights.resize(numClasses, WeightVec::Zero(featDimPx + 1)); // +1 for the bias
    m_pairwiseWeights.resize(numClasses * numClasses, WeightVec::Zero(2 * featDimPx + 1));
    m_higherOrderWeights.resize(numClasses * numClasses, WeightVec::Zero(2 * featDimCluster + 1));
    m_featureWeights.resize(numClasses * numClasses, WeightVec::Ones(featDimCluster));

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

Weights Weights::regularized() const
{
    Weights regularized = *this;
    for (size_t i = 0; i < m_featureWeights.size(); ++i)
        regularized.m_featureWeights[i] = (m_featureWeights[i] - Eigen::VectorXf::Ones(m_featureWeights[i].size()));
    return regularized;
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
        out.write("WEIGHT04", 8);
        uint32_t featDimPx = m_unaryWeights[0].size() - 1;
        uint32_t featDimCluster = m_featureWeights[0].size();
        uint32_t noUnaries = m_unaryWeights.size();
        uint32_t noPairwise = m_pairwiseWeights.size();
        uint32_t noHigherOrder = m_higherOrderWeights.size();
        uint32_t noFeature = m_featureWeights.size();
        out.write(reinterpret_cast<const char*>(&featDimPx), sizeof(featDimPx));
        out.write(reinterpret_cast<const char*>(&featDimCluster), sizeof(featDimCluster));
        out.write(reinterpret_cast<const char*>(&noUnaries), sizeof(noUnaries));
        out.write(reinterpret_cast<const char*>(&noPairwise), sizeof(noPairwise));
        out.write(reinterpret_cast<const char*>(&noHigherOrder), sizeof(noHigherOrder));
        out.write(reinterpret_cast<const char*>(&noFeature), sizeof(noFeature));
        for(auto const& e : m_unaryWeights)
        {
            assert(e.size() == featDimPx + 1);
            out.write(reinterpret_cast<const char*>(e.data()), sizeof(e(0)) * e.size());
        }
        for(auto const& e : m_pairwiseWeights)
        {
            assert(e.size() == featDimPx * 2 + 1);
            out.write(reinterpret_cast<const char*>(e.data()), sizeof(e(0)) * e.size());
        }
        for(auto const& e : m_higherOrderWeights)
        {
            assert(e.size() == featDimCluster * 2 + 1);
            out.write(reinterpret_cast<const char*>(e.data()), sizeof(e(0)) * e.size());
        }
        for(auto const& e : m_featureWeights)
        {
            assert(e.size() == featDimCluster);
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
        if(std::strncmp(id, "WEIGHT04", 8) != 0)
        {
            in.close();
            return false;
        }
        uint32_t featDimPx, featDimCluster, noUnaries, noPairwise, noHigherOrder, noFeature;
        in.read(reinterpret_cast<char*>(&featDimPx), sizeof(featDimPx));
        in.read(reinterpret_cast<char*>(&featDimCluster), sizeof(featDimCluster));
        in.read(reinterpret_cast<char*>(&noUnaries), sizeof(noUnaries));
        in.read(reinterpret_cast<char*>(&noPairwise), sizeof(noPairwise));
        in.read(reinterpret_cast<char*>(&noHigherOrder), sizeof(noHigherOrder));
        in.read(reinterpret_cast<char*>(&noFeature), sizeof(noFeature));
        m_unaryWeights.resize(noUnaries, WeightVec::Zero(featDimPx + 1));
        m_pairwiseWeights.resize(noPairwise, WeightVec::Zero(featDimPx * 2 + 1));
        m_higherOrderWeights.resize(noHigherOrder, WeightVec::Zero(featDimCluster * 2 + 1));
        m_featureWeights.resize(noFeature, WeightVec::Zero(featDimCluster));
        for(auto& e : m_unaryWeights)
            in.read(reinterpret_cast<char*>(e.data()), sizeof(e(0)) * (featDimPx + 1));
        for(auto& e : m_pairwiseWeights)
            in.read(reinterpret_cast<char*>(e.data()), sizeof(e(0)) * (featDimPx * 2 + 1));
        for(auto& e : m_higherOrderWeights)
            in.read(reinterpret_cast<char*>(e.data()), sizeof(e(0)) * (featDimCluster * 2 + 1));
        for(auto& e : m_featureWeights)
            in.read(reinterpret_cast<char*>(e.data()), sizeof(e(0)) * (featDimCluster));
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
    static const float threshold = 1e-3f;
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

WeightStats Weights::computeStats() const
{
    WeightStats stats;

    auto computeInit = [] (std::vector<WeightVec> const& w)
    {
        Stat s;
        s.mean = w[0].sum();
        s.max = w[0].maxCoeff();
        s.min = w[0].minCoeff();
        s.mag = w[0].squaredNorm();

        for(size_t i = 1; i < w.size(); ++i)
        {
            s.mean += w[i].sum();
            float m = w[i].maxCoeff();
            if(m > s.max)
                s.max = m;
            m = w[i].minCoeff();
            if(m < s.min)
                s.min = m;
            s.mag += w[i].squaredNorm();
        }
        return s;
    };

    auto stdevInit = [] (std::vector<WeightVec> const& w, float mean)
    {
        float sq_sum = 0;
        for(size_t i = 0; i < w.size(); ++i)
        {
            for(size_t j = 0; j < w[i].size(); ++j)
                sq_sum += std::pow(w[i](j) - mean, 2);
        }
        return sq_sum;
    };

    auto combine = [](WeightStats& wS)
    {
        wS.total.mean = wS.unary.mean + wS.pairwise.mean + wS.label.mean + wS.feature.mean;
        wS.total.stdev = wS.unary.stdev + wS.pairwise.stdev + wS.label.stdev + wS.feature.stdev;
        wS.total.mag = wS.unary.mag + wS.pairwise.mag + wS.label.mag + wS.feature.mag;
        wS.total.max = std::max({wS.unary.max, wS.pairwise.max, wS.label.max, wS.feature.max});
        wS.total.min = std::min({wS.unary.min, wS.pairwise.min, wS.label.min, wS.feature.min});
    };

    auto finish = [](Stat& s, size_t num)
    {
        s.mean /= num;
        s.stdev = std::sqrt(s.stdev / num);
        s.mag = std::sqrt(s.mag);
    };

    stats.unary = computeInit(m_unaryWeights);
    stats.pairwise = computeInit(m_pairwiseWeights);
    stats.label = computeInit(m_higherOrderWeights);
    stats.feature = computeInit(m_featureWeights);

    size_t numUnary = m_unaryWeights.size() * m_unaryWeights[0].size();
    size_t numPairwise = m_pairwiseWeights.size() * m_pairwiseWeights[0].size();
    size_t numLabelCon = m_higherOrderWeights.size() * m_higherOrderWeights[0].size();
    size_t numFeature = m_featureWeights.size() * m_featureWeights[0].size();

    stats.unary.stdev = stdevInit(m_unaryWeights, stats.unary.mean / numUnary);
    stats.pairwise.stdev = stdevInit(m_pairwiseWeights, stats.pairwise.mean / numPairwise);
    stats.label.stdev = stdevInit(m_higherOrderWeights, stats.label.mean / numLabelCon);
    stats.feature.stdev = stdevInit(m_featureWeights, stats.feature.mean / numFeature);

    combine(stats);

    finish(stats.unary, numUnary);
    finish(stats.pairwise, numPairwise);
    finish(stats.label, numLabelCon);
    finish(stats.feature, numFeature);
    finish(stats.total, numUnary + numPairwise + numLabelCon + numFeature);

    return stats;
}

void Weights::printStats(std::ostream& out) const
{
    auto printer = [&out] (std::string const& type, Stat const& s)
    {
        out << type << std::endl;
        out << " mean = " << s.mean << std::endl;
        out << " stdev = " << s.stdev << std::endl;
        out << " max = " << s.max << std::endl;
        out << " min = " << s.min << std::endl;
        out << " mag = " << s.mag << std::endl;
    };

    WeightStats wS = computeStats();

    printer("UNARY", wS.unary);
    printer("PAIRWISE", wS.pairwise);
    printer("LABELCON", wS.label);
    printer("FEATURE", wS.feature);
    printer("TOTAL", wS.total);
}
