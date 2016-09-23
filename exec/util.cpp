//
// Created by jan on 23.09.16.
//

#include <BaseProperties.h>
#include <Energy/WeightsVec.h>

PROPERTIES_DEFINE(Util,
                  GROUP_DEFINE(job,
                               PROP_DEFINE_A(std::string, showWeightFile, "", -sw)
                  )
)

bool showWeight(std::string const& weightFile)
{
    WeightsVec w(21ul);
    if (!w.read(weightFile))
    {
        std::cerr << "Couldn't read weight file \"" << weightFile << "\"" << std::endl;
        return false;
    }
    std::cout << "==========" << std::endl;
    std::cout << weightFile << ":" << std::endl;
    std::cout << w << std::endl;
    std::cout << "==========" << std::endl;
    return true;
}

int main(int argc, char** argv)
{
    UtilProperties properties;
    properties.read("properties/util.info");
    properties.fromCmd(argc, argv);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    if (!properties.job.showWeightFile.empty())
        showWeight(properties.job.showWeightFile);


    return 0;
}