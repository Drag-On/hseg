//
// Created by jan on 01.09.16.
//

#include <BaseProperties.h>
#include <Energy/WeightsVec.h>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

PROPERTIES_DEFINE(TrainDistMerge,
                  PROP_DEFINE(size_t, t, 0)
                  PROP_DEFINE(size_t, numClusters, 300)
                  PROP_DEFINE(float, learningRate, 1.f)
                  PROP_DEFINE(float, C, 1.f)
                  PROP_DEFINE(float, pairwiseSigmaSq, 0.05f)
                  PROP_DEFINE(std::string, weightFile, "")
                  PROP_DEFINE(std::string, in, "in/")
                  PROP_DEFINE(std::string, out, "out/weights.dat")
)

/**
 * Arguments:
 *  1 - Current iteration t
 *  2 - Output file
 *  3 - Input folder
 * @param argc
 * @param argv
 * @param properties
 */
void parseArguments(int argc, char* argv[], TrainDistMergeProperties& properties)
{
    if (argc > 1)
        properties.t = std::atoi(argv[1]);
    if (argc > 2)
        properties.out = std::string(argv[2]);
    if (argc > 3)
        properties.in = std::string(argv[3]);
}

int main(int argc, char* argv[])
{
    // Read properties
    TrainDistMergeProperties properties;
    properties.read("properties/training_dist_merge.info");
    parseArguments(argc, argv, properties);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    size_t const numClasses = 21;
    WeightsVec curWeights(numClasses, false);
    if(!curWeights.read(properties.weightFile))
    {
        std::cerr << "Couldn't read current weights from " << properties.weightFile << std::endl;
        return -1;
    }

    // Iterate over all files in the in directory and compute the sum
    WeightsVec sum(numClasses, false);
    boost::filesystem::path inPath(properties.in);
    size_t N = 0;
    for(auto& file : boost::make_iterator_range(boost::filesystem::directory_iterator(inPath), {}))
    {
        WeightsVec vec(numClasses, false);
        if(!vec.read(file.path().string()))
        {
            std::cerr << file.path() << " can not be read as a weights vector." << std::endl;
            continue;
        }
        sum += vec;
        N++;
    }
    sum *= properties.C / N;
    sum += curWeights;
    sum *= properties.learningRate / (properties.t + 1);
    curWeights -= sum;

    if(!curWeights.write(properties.out))
        std::cerr << "Couldn't write weights to file " << properties.out << std::endl;
    std::cout << curWeights << std::endl;

    return 0;
}