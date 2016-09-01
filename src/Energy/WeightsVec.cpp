//
// Created by jan on 23.08.16.
//

#include <fstream>
#include "Energy/WeightsVec.h"

WeightsVec::WeightsVec(Label numLabels, bool defaultInit)
        : m_numLabels(numLabels)
{
    if (defaultInit)
    {
        m_unaryWeights.resize(numLabels, 5.f);
        m_pairwiseWeights.resize((numLabels * numLabels) / 2, 300.f);
        m_featureWeights = {0.2f, 0.8f, 0.8f, 0.f};
        m_classWeight = 30.f;
    }
    else
    {
        m_unaryWeights.resize(numLabels, 0.f);
        m_pairwiseWeights.resize((numLabels * numLabels) / 2, 0.f);
        m_featureWeights = {0.f, 0.f, 0.f, 0.f};
        m_classWeight = 0.f;
    }
}

WeightsVec::WeightsVec(Label numLabels, float unaryWeight, float pairwiseWeight, float featA, float featB, float featC,
                 float featD, float labelWeight)
        : m_numLabels(numLabels)
{
    m_unaryWeights.resize(numLabels, unaryWeight);
    m_pairwiseWeights.resize((numLabels * numLabels) / 2, pairwiseWeight);
    m_featureWeights = {featA, featB, featC, featD};
    m_classWeight = labelWeight;
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
    return std::max(0.f, m_pairwiseWeights[index]);
}

Weight& WeightsVec::pairwise(Label l1, Label l2)
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

WeightsVec& WeightsVec::operator+=(WeightsVec const& other)
{
    assert(m_numLabels == other.m_numLabels);

    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] += other.m_unaryWeights[i];
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] += other.m_pairwiseWeights[i];
    m_featureWeights.m_a += other.m_featureWeights.m_a;
    m_featureWeights.m_b += other.m_featureWeights.m_b;
    m_featureWeights.m_c += other.m_featureWeights.m_c;
    m_featureWeights.m_d += other.m_featureWeights.m_d;
    m_classWeight += other.m_classWeight;

    return *this;
}

WeightsVec& WeightsVec::operator-=(WeightsVec const& other)
{
    assert(m_numLabels == other.m_numLabels);

    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] -= other.m_unaryWeights[i];
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] -= other.m_pairwiseWeights[i];
    m_featureWeights.m_a -= other.m_featureWeights.m_a;
    m_featureWeights.m_b -= other.m_featureWeights.m_b;
    m_featureWeights.m_c -= other.m_featureWeights.m_c;
    m_featureWeights.m_d -= other.m_featureWeights.m_d;
    m_classWeight -= other.m_classWeight;

    return *this;
}

WeightsVec& WeightsVec::operator*=(float factor)
{
    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] *= factor;
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] *= factor;
    m_featureWeights.m_a *= factor;
    m_featureWeights.m_b *= factor;
    m_featureWeights.m_c *= factor;
    m_featureWeights.m_d *= factor;
    m_classWeight *= factor;

    return *this;
}

WeightsVec& WeightsVec::operator*=(WeightsVec const& other)
{
    assert(m_numLabels == other.m_numLabels);

    for (size_t i = 0; i < m_unaryWeights.size(); ++i)
        m_unaryWeights[i] *= other.m_unaryWeights[i];
    for (size_t i = 0; i < m_pairwiseWeights.size(); ++i)
        m_pairwiseWeights[i] *= other.m_pairwiseWeights[i];
    m_featureWeights.m_a *= other.m_featureWeights.m_a;
    m_featureWeights.m_b *= other.m_featureWeights.m_b;
    m_featureWeights.m_c *= other.m_featureWeights.m_c;
    m_featureWeights.m_d *= other.m_featureWeights.m_d;
    m_classWeight *= other.m_classWeight;

    return *this;
}

Weight WeightsVec::sum() const
{
    return sumUnary() + sumPairwise() + sumSuperpixel();
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
    result += m_featureWeights.m_a;
    result += m_featureWeights.m_b;
    result += m_featureWeights.m_c;
    result += m_featureWeights.m_d;
    result += m_classWeight;
    return result;
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
    stream << "feature: " << weights.m_featureWeights.a() << ", " << weights.m_featureWeights.b() << ", "
           << weights.m_featureWeights.c() << ", " << weights.m_featureWeights.d() << std::endl;
    stream << "class: " << weights.m_classWeight;
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
        out.write(reinterpret_cast<const char*>(&m_featureWeights.m_a), sizeof(m_featureWeights.m_a));
        out.write(reinterpret_cast<const char*>(&m_featureWeights.m_b), sizeof(m_featureWeights.m_b));
        out.write(reinterpret_cast<const char*>(&m_featureWeights.m_c), sizeof(m_featureWeights.m_c));
        out.write(reinterpret_cast<const char*>(&m_featureWeights.m_d), sizeof(m_featureWeights.m_d));
        out.write(reinterpret_cast<const char*>(&m_classWeight), sizeof(m_classWeight));
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
        in.read(reinterpret_cast<char*>(&m_featureWeights.m_a), sizeof(m_featureWeights.m_a));
        in.read(reinterpret_cast<char*>(&m_featureWeights.m_b), sizeof(m_featureWeights.m_b));
        in.read(reinterpret_cast<char*>(&m_featureWeights.m_c), sizeof(m_featureWeights.m_c));
        in.read(reinterpret_cast<char*>(&m_featureWeights.m_d), sizeof(m_featureWeights.m_d));
        in.read(reinterpret_cast<char*>(&m_classWeight), sizeof(m_classWeight));
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

Weight WeightsVec::FeatureWeights::a() const
{
    return std::max(0.f, m_a);
}

Weight WeightsVec::FeatureWeights::b() const
{
    return std::max(0.f, m_b);
}

Weight WeightsVec::FeatureWeights::c() const
{
    return std::max(0.f, m_c);
}

Weight WeightsVec::FeatureWeights::d() const
{
    return std::max(0.f, m_d);
}

WeightsVec::FeatureWeights::FeatureWeights(Weight a, Weight b, Weight c, Weight d)
        : m_a(a),
          m_b(b),
          m_c(c),
          m_d(d)
{
}

void WeightsVec::FeatureWeights::set(Weight a, Weight b, Weight c, Weight d)
{
    m_a = a;
    m_b = b;
    m_c = c;
    m_d = d;
}
