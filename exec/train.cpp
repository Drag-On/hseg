//
// Created by jan on 29.08.16.
//

#include <BaseProperties.h>
#include <Energy/Weights.h>

PROPERTIES_DEFINE(Train,
                  PROP_DEFINE(size_t, numClusters, 300)
                  PROP_DEFINE(size_t, numIter, 100)
                  PROP_DEFINE(float, learningRate, 1.f)
                  PROP_DEFINE(float, C, 1.f)
                  PROP_DEFINE(float, pairwiseSigmaSq, 0.05f)
                  PROP_DEFINE(std::string, imageListFile, "")
                  PROP_DEFINE(std::string, groundTruthListFile, "")
                  PROP_DEFINE(std::string, unaryListFile, "")
                  PROP_DEFINE(std::string, imageBasePath, "")
                  PROP_DEFINE(std::string, groundTruthBasePath, "")
                  PROP_DEFINE(std::string, unaryBasePath, "")
                  PROP_DEFINE(std::string, outDir, "out/")
)

int main()
{
    // Read properties
    TrainProperties properties;
    properties.read("properties/training.info");
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    size_t const numClasses = 21;
    Weights weights(numClasses, false); // Zero-weights

    return 0;
}