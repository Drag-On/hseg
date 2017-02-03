//
// Created by jan on 24.01.17.
//

#include "helper/clustering_helper.h"

namespace helper
{
    namespace clustering
    {
        bool write(std::string const& filename, LabelImage const& clustering, std::vector<Cluster> const& clusters)
        {
            std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
            if(out.is_open())
            {
                out.write("CLUSTE01", 8);
                uint32_t numClusters = clusters.size();
                uint32_t featDim = clusters.front().m_feature.size();
                uint32_t width = clustering.width();
                uint32_t height = clustering.height();
                out.write(reinterpret_cast<const char*>(&numClusters), sizeof(numClusters));
                out.write(reinterpret_cast<const char*>(&featDim), sizeof(featDim));
                out.write(reinterpret_cast<const char*>(&width), sizeof(width));
                out.write(reinterpret_cast<const char*>(&height), sizeof(height));
                out.write(reinterpret_cast<const char*>(clustering.data().data()), sizeof(Label) * clustering.pixels());
                for(auto const& c : clusters)
                {
                    out.write(reinterpret_cast<const char*>(&c.m_label), sizeof(Label));
                    out.write(reinterpret_cast<const char*>(c.m_feature.data()), sizeof(float) * featDim);
                }
                out.close();
                return true;
            }
            return false;
        }

        bool read(std::string const& filename, LabelImage& outClustering, std::vector<Cluster>& outClusters)
        {
            std::ifstream in(filename, std::ios::in | std::ios::binary);
            if(in.is_open())
            {
                char id[8];
                in.read(id, 8);
                if (std::strncmp(id, "CLUSTE01", 8) != 0)
                {
                    in.close();
                    return false;
                }
                uint32_t numClusters, featDim, width, height;
                in.read(reinterpret_cast<char*>(&numClusters), sizeof(numClusters));
                in.read(reinterpret_cast<char*>(&featDim), sizeof(featDim));
                in.read(reinterpret_cast<char*>(&width), sizeof(width));
                in.read(reinterpret_cast<char*>(&height), sizeof(height));

                outClustering = LabelImage(width, height);
                in.read(reinterpret_cast<char*>(outClustering.data().data()), sizeof(Label) * outClustering.pixels());

                for(uint32_t k = 0; k < numClusters; ++k)
                {
                    Cluster c;
                    c.m_feature.resize(featDim);
                    in.read(reinterpret_cast<char*>(&c.m_label), sizeof(Label));
                    in.read(reinterpret_cast<char*>(c.m_feature.data()), sizeof(float) * featDim);
                    outClusters.push_back(c);
                }
                in.close();
                return true;
            }
            return false;
        }
    }
}