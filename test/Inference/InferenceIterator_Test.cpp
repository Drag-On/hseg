//
// Created by jan on 24.01.17.
//

#include <gtest/gtest.h>
#include <Inference/InferenceIterator.h>

class TestEnergyFunction
{
public:
    inline Cost higherOrderCost(Feature const& f1, Feature const& f2, Label l1, Label l2) const
    {
        return l1 == l2 ? 0 : 1;
    }

    inline Cost featureCost(Feature const& f1, Feature const& f2) const
    {
        return std::abs((f1 - f2)(0));
    }
};

class TestFeatureImage : public FeatureImage
{
public:
    TestFeatureImage()
    {
        m_width = m_height = 2;
        m_dim = 1;
        m_features.resize(4, Feature::Ones(1));
        m_features[0] << 0;
        m_features[1] << 1;
        m_features[2] << 0.4;
        m_features[3] << 2;
    }
};

class TestInferenceIterator : public InferenceIterator<TestEnergyFunction>
{
private:
    TestFeatureImage m_feat;
public:
    TestInferenceIterator()
            : InferenceIterator<TestEnergyFunction>(new TestEnergyFunction, &m_feat)
    {
    }

    ~TestInferenceIterator()
    {
        delete m_pEnergy;
    }

    void testUpdateClusterAffiliation()
    {
        LabelImage outClustering(2, 2);
        LabelImage labeling(2, 2);
        labeling.at(1, 0) = 1;
        labeling.at(1, 1) = 1;
        std::vector<Cluster> clusters(2);
        clusters[0].m_label = 0;
        clusters[0].m_feature = Feature::Ones(1);
        clusters[0].m_feature << 1.f;
        clusters[1].m_label = 1;
        clusters[1].m_feature = Feature::Ones(1);
        clusters[1].m_feature << 3.f;

        updateClusterAffiliation(outClustering, labeling, clusters);

        EXPECT_EQ(0, outClustering.at(0, 0));
        EXPECT_EQ(0, outClustering.at(1, 0));
        EXPECT_EQ(0, outClustering.at(0, 1));
        EXPECT_EQ(1, outClustering.at(1, 1));
    }
};

TEST(InferenceIterator,updateClusterAffiliation)
{
    TestInferenceIterator t;
    t.testUpdateClusterAffiliation();
}