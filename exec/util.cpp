//
// Created by jan on 23.09.16.
//

#include <BaseProperties.h>
#include <Energy/WeightsVec.h>

PROPERTIES_DEFINE(Util,
                  GROUP_DEFINE(job,
                               PROP_DEFINE_A(std::string, showWeightFile, "", -sw)
                               PROP_DEFINE_A(std::string, writeWeightFile, "", -ww)
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

bool writeWeight(std::string const& weightFile)
{
    std::cout << "==========" << std::endl;
    std::cout << "Unary weight: ";
    float u = 0;
    std::cin >> u;
    std::cout << "Pairwise weight: ";
    float p = 0;
    std::cin >> p;
    std::cout << "Color weight: ";
    float c = 0;
    std::cin >> c;
    std::cout << "Spatial weight: ";
    float s = 0;
    std::cin >> s;
    std::cout << "Class weight: ";
    float l = 0;
    std::cin >> l;
    std::cout << "==========" << std::endl;
    WeightsVec w(21ul, u, p, c, s, s, 0, l);
    return w.write(weightFile);
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

    if(!properties.job.writeWeightFile.empty())
        writeWeight(properties.job.writeWeightFile);


    return 0;
}