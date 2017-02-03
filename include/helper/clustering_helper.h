//
// Created by jan on 24.01.17.
//

#ifndef HSEG_CLUSTERING_HELPER_H
#define HSEG_CLUSTERING_HELPER_H

#include <Inference/Cluster.h>
#include <Image/Image.h>
#include <fstream>

namespace helper
{
    namespace clustering
    {
        /**
         * Writes clustering information to file
         * @param filename File to write to
         * @param clustering Cluster affiliation map
         * @param clusters List of clusters including their labels and features
         * @return True in case the file was sucessfully written, otherwise false
         */
        bool write(std::string const& filename, LabelImage const& clustering, std::vector<Cluster> const& clusters);

        /**
         * Reads in clustering information from file
         * @param filename File to read form
         * @param outClustering Cluster affiliation map to write to
         * @param outClusters List of clusters to write to
         * @return True in case of success, otherwise false
         */
        bool read(std::string const& filename, LabelImage& outClustering, std::vector<Cluster>& outClusters);
    }
}

#endif //HSEG_CLUSTERING_HELPER_H
