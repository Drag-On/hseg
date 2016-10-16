//
// Created by jan on 23.09.16.
//

#include <BaseProperties.h>
#include <Energy/WeightsVec.h>
#include <helper/image_helper.h>
#include <helper/coordinate_helper.h>
#include <boost/filesystem/path.hpp>

PROPERTIES_DEFINE(Util,
                  GROUP_DEFINE(job,
                               PROP_DEFINE_A(std::string, showWeightFile, "", -sw)
                               PROP_DEFINE_A(std::string, writeWeightFile, "", -ww)
                               PROP_DEFINE_A(std::string, fillGroundTruth, "", -f)
                  )
                  GROUP_DEFINE(FillGroundTruth,
                               PROP_DEFINE(size_t, numClasses, 21u)
                               PROP_DEFINE(std::string, outDir, "")
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

bool fillGroundTruth(UtilProperties const& properties)
{
    RGBImage gtRGB;
    if(!gtRGB.read(properties.job.fillGroundTruth))
    {
        std::cerr << "Couldn't open ground truth image from \"" << properties.job.fillGroundTruth << "\"." << std::endl;
        return false;
    }
    auto cmap = helper::image::generateColorMapVOC(256);
    LabelImage gt = helper::image::decolorize(gtRGB, cmap);
    LabelImage fixedGt = gt;

    // Find invalid pixels
    size_t const numClasses = properties.FillGroundTruth.numClasses;
    std::vector<size_t> invalid;
    for(size_t i = 0; i < gt.pixels(); ++i)
        if(gt.atSite(i) >= numClasses)
            invalid.push_back(i);

    // Fix the invalid pixels iteratively
    while(!invalid.empty())
    {
        LabelImage curFixed = fixedGt;
        for(auto iter = invalid.begin(); iter != invalid.end(); )
        {
            // Check most prevalent adjacent label
            std::map<Label, size_t> adjacentLabels;
            auto coords = helper::coord::siteTo2DCoordinate(*iter, gt.width());
            if(coords.x() > 0)
            {
                Label leftLabel = curFixed.at(coords.x() - 1, coords.y());
                if(leftLabel < numClasses)
                    adjacentLabels[leftLabel]++;
            }
            if(coords.x() < gt.width() - 1)
            {
                Label rightLabel = curFixed.at(coords.x() + 1, coords.y());
                if(rightLabel < numClasses)
                    adjacentLabels[rightLabel]++;
            }
            if(coords.y() > 0)
            {
                Label upperLabel = curFixed.at(coords.x(), coords.y() - 1);
                if(upperLabel < numClasses)
                    adjacentLabels[upperLabel]++;
            }
            if(coords.y() < gt.height() - 1)
            {
                Label lowerLabel = curFixed.at(coords.x(), coords.y() + 1);
                if(lowerLabel < numClasses)
                    adjacentLabels[lowerLabel]++;
            }
            if(!adjacentLabels.empty())
            {
                auto max = std::max_element(adjacentLabels.begin(), adjacentLabels.end(),
                                            [](const std::pair<Label, size_t>& p1,
                                               const std::pair<Label, size_t>& p2) { return p1.second < p2.second; });
                Label newLabel = max->first;
                fixedGt.atSite(*iter) = newLabel;
                iter = invalid.erase(iter);
            }
            else
                ++iter;
        }
    }

    // Write result back to file
    RGBImage result = helper::image::colorize(fixedGt, cmap);
    cv::Mat resultMat = static_cast<cv::Mat>(result);
    std::string filename = boost::filesystem::path(properties.job.fillGroundTruth).filename().string();
    if(!cv::imwrite(properties.FillGroundTruth.outDir + filename, resultMat))
    {
        std::cerr << "Couldn't write result to \"" << properties.FillGroundTruth.outDir + filename << "\"." << std::endl;
        return false;
    }
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

    if(!properties.job.writeWeightFile.empty())
        writeWeight(properties.job.writeWeightFile);

    if(!properties.job.fillGroundTruth.empty())
        fillGroundTruth(properties);

    return 0;
}