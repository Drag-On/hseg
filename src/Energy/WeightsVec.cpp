//
// Created by jan on 23.08.16.
//

#include <fstream>
#include <Eigen/Eigenvalues>
#include "Energy/WeightsVec.h"

WeightsVec::WeightsVec(Label numLabels, bool defaultInit)
        : m_numLabels(numLabels)
{
    if (defaultInit)
    {
        m_unaryWeights.resize(numLabels, 5.f);
        m_pairwiseWeights.resize((numLabels * numLabels) / 2, 300.f);
        m_featureWeights.resize((m_numFeatures + 1) * m_numFeatures / 2, 0.f);
        m_classWeights.resize((numLabels * numLabels) / 2, 30.f);
        featureWeight(0, 0) = featureWeight(1, 1) = featureWeight(2, 2) = 0.85f;
        featureWeight(3, 3) = featureWeight(4, 4) = 0.15f;
    }
    else
    {
        m_unaryWeights.resize(numLabels, 0.f);
        m_pairwiseWeights.resize((numLabels * numLabels) / 2, 0.f);
        m_featureWeights.resize((m_numFeatures + 1) * m_numFeatures / 2, 0.f);
        m_classWeights.resize((numLabels * numLabels) / 2, 0.f);
        featureWeight(0, 0) = featureWeight(1, 1) = featureWeight(2, 2) = featureWeight(3, 3) = featureWeight(4, 4) = 1.f;
    }
}

WeightsVec::WeightsVec(Label numLabels, Weight unary, Weight pairwise, Weight feature, Weight label)
        : m_numLabels(numLabels)
{
    m_unaryWeights.resize(numLabels, unary);
    m_pairwiseWeights.resize((numLabels * numLabels) / 2, pairwise);
    m_featureWeights.resize((m_numFeatures + 1) * m_numFeatures / 2, 0.f);
    m_classWeights.resize((numLabels * numLabels) / 2, label);
    featureWeight(0, 0) = featureWeight(1, 1) = featureWeight(2, 2) = featureWeight(3, 3) = featureWeight(4, 4) = feature;
}

Weight WeightsVec::pairwise(Label l1, Label l2) const
{
    // Diagonal is always zero
    if (l1 == l2)
        return 0;

    // The weight l1->l2 is the same as l2->l1
    if (l2 < l1)
        std::swap(l1, l2);

    // Pairwise indices are stored as upper triangular matrix
    size_t const index = l1 + l2 * (l2 - 1) / 2;
    assert(index < m_pairwiseWeights.size());
    return std::max<Weight>(0.f, m_pairwiseWeights[index]);
}

Weight WeightsVec::classWeight(Label l1, Label l2) const
{
    // Diagonal is always zero
    if (l1 == l2)
        return 0;

    // The weight l1->l2 is the same as l2->l1
    if (l2 < l1)
        std::swap(l1, l2);

    // Pairwise indices are stored as upper triangular matrix
    size_t const index = l1 + l2 * (l2 - 1) / 2;
    assert(index < m_classWeights.size());
    return std::max<Weight>(0.f, m_classWeights[index]);
}

float& WeightsVec::pairwise(Label l1, Label l2)
{
    // Diagonal is always zero
    assert(l1 != l2);

    // The weight l1->l2 is the same as l2->l1
    if (l2 < l1)
        std::swap(l1, l2);

    // Pairwise indices are stored as upper triangular matrix
    size_t const index = l1 + l2 * (l2 - 1) / 2;
    assert(index < m_pairwiseWeights.size());
    return m_pairwiseWeights[index];
}

float& WeightsVec::classWeight(Label l1, Label l2)
{
    // Diagonal is always zero
    assert(l1 != l2);

    // The weight l1->l2 is the same as l2->l1
    if (l2 < l1)
        std::swap(l1, l2);

    // Pairwise indices are stored as upper triangular matrix
    size_t const index = l1 + l2 * (l2 - 1) / 2;
    assert(index < m_classWeights.size());
    return m_classWeights[index];
}

WeightsVec& WeightsVec::operator+=(WeightsVec const& other)
{
    assert(m_numLabels == other.m_numLabels);

    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] += other.m_unaryWeights[i];
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] += other.m_pairwiseWeights[i];
    for (size_t i = 0; i < m_featureWeights.size(); ++i)
        m_featureWeights[i] += other.m_featureWeights[i];
    for (size_t i = 0; i < m_classWeights.size(); ++i)
        m_classWeights[i] += other.m_classWeights[i];

    return *this;
}

WeightsVec& WeightsVec::operator-=(WeightsVec const& other)
{
    assert(m_numLabels == other.m_numLabels);

    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] -= other.m_unaryWeights[i];
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] -= other.m_pairwiseWeights[i];
    for (size_t i = 0; i < m_featureWeights.size(); ++i)
        m_featureWeights[i] -= other.m_featureWeights[i];
    for (size_t i = 0; i < m_classWeights.size(); ++i)
        m_classWeights[i] -= other.m_classWeights[i];

    return *this;
}

WeightsVec& WeightsVec::operator*=(float factor)
{
    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] *= factor;
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] *= factor;
    for (size_t i = 0; i < m_featureWeights.size(); ++i)
        m_featureWeights[i] *= factor;
    for (size_t i = 0; i < m_classWeights.size(); ++i)
        m_classWeights[i] *= factor;

    return *this;
}

WeightsVec& WeightsVec::operator*=(WeightsVec const& other)
{
    assert(m_numLabels == other.m_numLabels);

    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] *= other.m_unaryWeights[i];
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] *= other.m_pairwiseWeights[i];
    for (size_t i = 0; i < m_featureWeights.size(); ++i)
        m_featureWeights[i] *= other.m_featureWeights[i];
    for (size_t i = 0; i < m_classWeights.size(); ++i)
        m_classWeights[i] *= other.m_classWeights[i];

    return *this;
}

Weight WeightsVec::sumUnary() const
{
    Weight result = 0;
    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        result += m_unaryWeights[i];
    return result;
}

Weight WeightsVec::sumPairwise() const
{
    Weight result = 0;
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        result += m_pairwiseWeights[i];
    return result;
}

Weight WeightsVec::sumSuperpixel() const
{
    Weight result = 0;
    for (size_t i = 0; i < m_classWeights.size(); ++i)
        result += m_classWeights[i];
    for (size_t i = 0; i < m_featureWeights.size(); ++i)
        result += m_featureWeights[i];
    return result;
}

Weight WeightsVec::sqNorm() const
{
    Weight sqNorm = 0;

    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        sqNorm += m_unaryWeights[i] * m_unaryWeights[i];
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        sqNorm += m_pairwiseWeights[i] * m_pairwiseWeights[i];
    for (size_t i = 0; i < m_featureWeights.size(); ++i)
        sqNorm += m_featureWeights[i] * m_featureWeights[i];
    for (size_t i = 0; i < m_classWeights.size(); ++i)
        sqNorm += m_classWeights[i] * m_classWeights[i];

    return sqNorm;
}

std::ostream& operator<<(std::ostream& stream, WeightsVec const& weights)
{
    stream << "unary: ";
    for (size_t i = 0; i < weights.m_unaryWeights.size() - 1; ++i)
        stream << weights.m_unaryWeights[i] << ", ";
    if (!weights.m_unaryWeights.empty())
        stream << weights.m_unaryWeights.back() << std::endl;
    stream << "pairwise: ";
    for (size_t i = 0; i < weights.m_pairwiseWeights.size() - 1; ++i)
        stream << weights.m_pairwiseWeights[i] << ", ";
    if (!weights.m_pairwiseWeights.empty())
        stream << weights.m_pairwiseWeights.back() << std::endl;
    stream << "feature: ";
    for (size_t i = 0; i < weights.m_featureWeights.size() - 1; ++i)
        stream << weights.m_featureWeights[i] << ", ";
    if(!weights.m_featureWeights.empty())
        stream << weights.m_featureWeights.back() << std::endl;
    stream << "class: ";
    for (size_t i = 0; i < weights.m_classWeights.size() - 1; ++i)
        stream << weights.m_classWeights[i] << ", ";
    if (!weights.m_classWeights.empty())
        stream << weights.m_classWeights.back() << std::endl;
    return stream;
}

bool WeightsVec::write(std::string const& filename) const
{
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    if(out.is_open())
    {
        out.write("WEIGHT", 6);
        size_t noUnaries = m_unaryWeights.size();
        out.write(reinterpret_cast<const char*>(&noUnaries), sizeof(noUnaries));
        out.write(reinterpret_cast<const char*>(m_unaryWeights.data()), sizeof(m_unaryWeights[0]) * noUnaries);
        size_t noPairwise = m_pairwiseWeights.size();
        out.write(reinterpret_cast<const char*>(&noPairwise), sizeof(noPairwise));
        out.write(reinterpret_cast<const char*>(m_pairwiseWeights.data()), sizeof(m_pairwiseWeights[0]) * noPairwise);
        size_t noFeature = m_featureWeights.size();
        out.write(reinterpret_cast<const char*>(&noFeature), sizeof(noFeature));
        out.write(reinterpret_cast<const char*>(m_featureWeights.data()), sizeof(m_featureWeights[0]) * noFeature);
        size_t noClass = m_classWeights.size();
        out.write(reinterpret_cast<const char*>(&noClass), sizeof(noClass));
        out.write(reinterpret_cast<const char*>(m_classWeights.data()), sizeof(m_classWeights[0]) * noClass);
        out.close();
        return true;
    }
    return false;
}

bool WeightsVec::read(std::string const& filename)
{
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    if(in.is_open())
    {
        char id[6];
        in.read(id, 6);
        if(std::strncmp(id, "WEIGHT", 6) != 0)
        {
            in.close();
            return false;
        }
        size_t noUnaries;
        in.read(reinterpret_cast<char*>(&noUnaries), sizeof(noUnaries));
        m_unaryWeights.resize(noUnaries);
        in.read(reinterpret_cast<char*>(m_unaryWeights.data()), sizeof(m_unaryWeights[0]) * noUnaries);
        size_t noPairwise;
        in.read(reinterpret_cast<char*>(&noPairwise), sizeof(noPairwise));
        m_pairwiseWeights.resize(noPairwise);
        in.read(reinterpret_cast<char*>(m_pairwiseWeights.data()), sizeof(m_pairwiseWeights[0]) * noPairwise);
        size_t noFeature;
        in.read(reinterpret_cast<char*>(&noFeature), sizeof(noFeature));
        m_featureWeights.resize(noFeature);
        in.read(reinterpret_cast<char*>(m_featureWeights.data()), sizeof(m_featureWeights[0]) * noFeature);
        size_t noClass;
        in.read(reinterpret_cast<char*>(&noClass), sizeof(noClass));
        m_classWeights.resize(noClass);
        in.read(reinterpret_cast<char*>(m_classWeights.data()), sizeof(m_classWeights[0]) * noClass);
        in.close();
        return true;
    }
    return false;
}

bool WeightsVec::hasPairwiseWeight() const
{
    bool hasPairwise = false;
    for(auto const& e : m_pairwiseWeights)
    {
        if(std::round(e) > 0)
            return true;
    }
    return hasPairwise;
}

std::vector<float>& WeightsVec::unaryWeights()
{
    return m_unaryWeights;
}

std::vector<float>& WeightsVec::pairwiseWeights()
{
    return m_pairwiseWeights;
}

std::vector<float>& WeightsVec::classWeights()
{
    return m_classWeights;
}

void WeightsVec::clampToFeasible()
{
    // All values are permitted for unary weights
    // Pairwise must be positive
    for(auto& e : m_pairwiseWeights)
        e = std::max<Weight>(0.f, e);
    // Class weights must be positive
    for(auto& e : m_classWeights)
        e = std::max<Weight>(0.f, e);
    // Feature weights must be positive semi-definite
    Matrix5 featureWeights = feature();
    Eigen::SelfAdjointEigenSolver<Matrix5> es(featureWeights);
    Matrix5 D = es.eigenvalues().cast<float>().asDiagonal();
    Matrix5 V = es.eigenvectors().cast<float>();
    for(uint16_t i = 0; i < es.eigenvalues().size(); ++i)
        if(D(i, i) < 0)
            D(0, 0) = 0;
    Matrix5 fixedFeatureWeights = V * D * V.inverse();
    for (uint16_t i = 0; i < m_numFeatures; ++i)
    {
        for(uint16_t j = i; j < m_numFeatures; ++j)
        {
            assert(i + (j + 1) * j / 2 < m_featureWeights.size());
            m_featureWeights[i + (j + 1) * j / 2] = fixedFeatureWeights(i, j);
        }
    }
}
