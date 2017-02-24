//
// Created by jan on 23.09.16.
//

#include <BaseProperties.h>
#include <Energy/Weights.h>
#include <helper/image_helper.h>
#include <helper/coordinate_helper.h>
#include <boost/filesystem/path.hpp>
#include <Energy/LossAugmentedEnergyFunction.h>
#include <boost/filesystem/operations.hpp>

PROPERTIES_DEFINE(Util,
                  GROUP_DEFINE(job,
                               PROP_DEFINE_A(std::string, showWeightFile, "", --show)
                               PROP_DEFINE_A(std::string, writeWeightFileText, "", --toText)
                               PROP_DEFINE_A(std::string, fillGroundTruth, "", --fill_gt)
                               PROP_DEFINE_A(std::string, pairwiseStatistics, "", --pair_stats)
                               PROP_DEFINE_A(std::string, maxLoss, "", --max_loss)
                               PROP_DEFINE_A(std::string, outline, "", --outline)
                               PROP_DEFINE_A(std::string, rescale, "", --rescale)
                               PROP_DEFINE_A(std::string, scaleUp, "", --scale_up)
                               PROP_DEFINE_A(std::string, matchGt, "", --match_gt)
                               PROP_DEFINE_A(std::string, copyFixPNG, "", --fix_PNG)
                  )
                  GROUP_DEFINE(dataset,
                               PROP_DEFINE_A(std::string, list, "", -l)
                               GROUP_DEFINE(path,
                                            PROP_DEFINE_A(std::string, img, "", --img)
                                            PROP_DEFINE_A(std::string, gt, "", --gt)
                                            PROP_DEFINE_A(std::string, rgb, "", --rgb)
                               )
                               GROUP_DEFINE(extension,
                                            PROP_DEFINE_A(std::string, img, ".mat", --img_ext)
                                            PROP_DEFINE_A(std::string, gt, ".png", --gt_ext)
                                            PROP_DEFINE_A(std::string, rgb, ".jpg", --rgb_ext)
                               )
                               GROUP_DEFINE(constants,
                                            PROP_DEFINE_A(uint32_t, numClasses, 21, --numClasses)
                                            PROP_DEFINE_A(uint32_t, featDim, 512, --featDim)
                               )
                  )
                  GROUP_DEFINE(param,
                               PROP_DEFINE_A(ClusterId, numClusters, 100, --numClusters)
                  )
                  PROP_DEFINE_A(std::string, in, "", -i)
                  PROP_DEFINE_A(std::string, out, "", -o)
                  PROP_DEFINE_A(float, rescaleFactor, 0.5f, --rescale)
                  PROP_DEFINE_A(ARG(std::array<unsigned short, 3>), border, ARG(std::array<unsigned short, 3>{255, 255, 255}), --color)
)

bool showWeight(std::string const& weightFile, UtilProperties const& properties)
{
    Weights w(properties.dataset.constants.numClasses, properties.dataset.constants.featDim);
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

bool writeWeightFileText(UtilProperties const& properties)
{
    size_t const numClasses = properties.dataset.constants.numClasses;
    size_t const featDim = properties.dataset.constants.featDim;
    Weights weightsVec(numClasses, featDim);
    if (!weightsVec.read(properties.job.writeWeightFileText))
    {
        std::cerr << "Couldn't read weight file \"" << properties.job.writeWeightFileText << "\"" << std::endl;
        return false;
    }
    Weights const& w = weightsVec;

    std::ofstream unaryOut(properties.out + "weights.unary.csv");
    if (unaryOut.is_open())
    {
        unaryOut << w.unary(0);
        for (size_t i = 1; i < numClasses; ++i)
            unaryOut << "\t" << w.unary(i);
        unaryOut << std::endl;
        unaryOut.close();
    }
    else
        std::cerr << "Couldn't write unary weights to \"" << properties.out << "weights.unary.csv" << "\""
                  << std::endl;

    std::ofstream pairwiseOut(properties.out + "weights.pairwise.csv");
    if (pairwiseOut.is_open())
    {
        for (Label l1 = 0; l1 < numClasses; ++l1)
        {
            pairwiseOut << w.pairwise(l1, 0);
            for (Label l2 = 1; l2 < numClasses; ++l2)
                pairwiseOut << "\t" << w.pairwise(l1, l2);
            pairwiseOut << std::endl;
        }
        pairwiseOut.close();
    }
    else
        std::cerr << "Couldn't write pairwise weights to \"" << properties.out << "weights.pairwise.csv" << "\""
                  << std::endl;

    return true;
}

bool fillGroundTruth(UtilProperties const& properties)
{
    LabelImage gt;
    auto errCode = helper::image::readPalettePNG(properties.job.fillGroundTruth, gt, nullptr);
    if (errCode != helper::image::PNGError::Okay)
    {
        std::cerr << "Couldn't open ground truth image from \"" << properties.job.fillGroundTruth << "\". Error Code: " << (int) errCode << std::endl;
        return false;
    }
    auto cmap = helper::image::generateColorMapVOC(256);
    LabelImage fixedGt = gt;

    // Find invalid pixels
    size_t const numClasses = properties.dataset.constants.numClasses;
    std::vector<size_t> invalid;
    for (size_t i = 0; i < gt.pixels(); ++i)
        if (gt.atSite(i) >= numClasses)
            invalid.push_back(i);

    // Fix the invalid pixels iteratively
    while (!invalid.empty())
    {
        LabelImage curFixed = fixedGt;
        for (auto iter = invalid.begin(); iter != invalid.end();)
        {
            // Check most prevalent adjacent label
            std::map<Label, size_t> adjacentLabels;
            auto coords = helper::coord::siteTo2DCoordinate(*iter, gt.width());
            if (coords.x() > 0)
            {
                Label leftLabel = curFixed.at(coords.x() - 1, coords.y());
                if (leftLabel < numClasses)
                    adjacentLabels[leftLabel]++;
            }
            if (coords.x() < gt.width() - 1)
            {
                Label rightLabel = curFixed.at(coords.x() + 1, coords.y());
                if (rightLabel < numClasses)
                    adjacentLabels[rightLabel]++;
            }
            if (coords.y() > 0)
            {
                Label upperLabel = curFixed.at(coords.x(), coords.y() - 1);
                if (upperLabel < numClasses)
                    adjacentLabels[upperLabel]++;
            }
            if (coords.y() < gt.height() - 1)
            {
                Label lowerLabel = curFixed.at(coords.x(), coords.y() + 1);
                if (lowerLabel < numClasses)
                    adjacentLabels[lowerLabel]++;
            }
            if (!adjacentLabels.empty())
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
    std::string filename = boost::filesystem::path(properties.job.fillGroundTruth).filename().string();
    errCode = helper::image::writePalettePNG(properties.out + filename, fixedGt, cmap);
    if (errCode != helper::image::PNGError::Okay)
    {
        std::cerr << "Couldn't write result to \"" << properties.out + filename << "\". Error Code: " << (int) errCode << std::endl;
        return false;
    }
    return true;
}

std::vector<std::string> readLines(std::string filename)
{
    std::vector<std::string> list;
    std::ifstream in(filename, std::ios::in);
    if (in.is_open())
    {
        std::string line;
        while (std::getline(in, line))
            list.push_back(line);
        in.close();
    }
    return list;
}

bool pairwiseStatistics(UtilProperties const& properties)
{
    // Read in files to consider
    std::vector<std::string> list = readLines(properties.job.pairwiseStatistics);

    auto cmap = helper::image::generateColorMapVOC(256);

    size_t pairwiseConnections = 0;
    std::vector<float> pairwiseWeights((properties.dataset.constants.numClasses * properties.dataset.constants.numClasses) / 2, 0.f);

    // Iterate them
    for (auto const& s : list)
    {
        std::string imageFile = properties.dataset.path.img + s + properties.dataset.extension.img;
        std::string gtFile = properties.dataset.path.gt + s + properties.dataset.extension.gt;
        RGBImage image, gtRGB;
        if (!image.read(imageFile))
        {
            std::cerr << "Couldn't read color image \"" << imageFile << "\"." << std::endl;
            return false;
        }
        if (!gtRGB.read(gtFile))
        {
            std::cerr << "Couldn't read ground truth image \"" << gtFile << "\"." << std::endl;
            return false;
        }
        LabelImage gt = helper::image::decolorize(gtRGB, cmap);

        for (size_t i = 0; i < gt.pixels(); ++i)
        {
            size_t l = gt.atSite(i);
            auto coords = helper::coord::siteTo2DCoordinate(i, gt.width());
            decltype(coords) coordsR = {static_cast<Coord>(coords.x() + 1), coords.y()};
            decltype(coords) coordsD = {coords.x(), static_cast<Coord>(coords.y() + 1)};
            if (coordsR.x() < gt.width())
            {
                size_t siteR = helper::coord::coordinateToSite(coordsR.x(), coordsR.y(), gt.width());
                size_t lR = gt.atSite(siteR);
                pairwiseConnections++;
                if (lR < l)
                    std::swap(l, lR);
                if (l != lR)
                    pairwiseWeights[l + lR * (lR - 1) / 2]++;
            }
            if (coordsD.y() < gt.height())
            {
                size_t siteD = helper::coord::coordinateToSite(coordsD.x(), coordsD.y(), gt.width());
                size_t lD = gt.atSite(siteD);
                pairwiseConnections++;
                if (lD < l)
                    std::swap(l, lD);
                if (l != lD)
                    pairwiseWeights[l + lD * (lD - 1) / 2]++;
            }
        }
    }

    for (size_t i = 0; i < pairwiseWeights.size(); ++i)
        pairwiseWeights[i] /= pairwiseConnections;

    for (size_t l1 = 0; l1 < properties.dataset.constants.numClasses; ++l1)
    {
        for (size_t l2 = 0; l2 < properties.dataset.constants.numClasses; ++l2)
        {
            if (l1 == l2)
            {
                std::cout << 0 << "\t";
                continue;
            }
            if (l2 < l1)
                std::cout << pairwiseWeights[l2 + l1 * (l1 - 1) / 2] << "\t";
            else
                std::cout << pairwiseWeights[l1 + l2 * (l2 - 1) / 2] << "\t";
        }
        std::cout << std::endl;
    }

    return true;
}

bool computeMaxLoss(UtilProperties const& properties)
{
    // Read in files to consider
    std::vector<std::string> list = readLines(properties.job.maxLoss);

    auto cmap = helper::image::generateColorMapVOC(256);

    // Iterate them
    float maxLoss = 0.f;
    for (auto const& s : list)
    {
        std::string imageFile = properties.dataset.path.img + s + properties.dataset.extension.img;
        std::string gtFile = properties.dataset.path.gt + s + properties.dataset.extension.gt;
        RGBImage image, gtRGB;
        if (!image.read(imageFile))
        {
            std::cerr << "Couldn't read color image \"" << imageFile << "\"." << std::endl;
            return false;
        }
        if (!gtRGB.read(gtFile))
        {
            std::cerr << "Couldn't read ground truth image \"" << gtFile << "\"." << std::endl;
            return false;
        }
        LabelImage gt = helper::image::decolorize(gtRGB, cmap);

        // Compute loss factor for this image
        float lossFactor = LossAugmentedEnergyFunction::computeLossFactor(gt, properties.dataset.constants.numClasses);

        maxLoss += image.pixels() * lossFactor;
    }
    maxLoss /= list.size();

    std::cout << "Maximum loss from set \"" << properties.job.maxLoss << "\" (" << list.size() << " images): "
              << maxLoss << std::endl;
    return true;
}

bool outline(UtilProperties const& /*properties*/)
{
    // Load color and superpixel image
//    std::string filename = boost::filesystem::path(properties.job.outline).stem().string();
//    RGBImage color, spRGB;
//    if (!color.read(properties.dataset.path.img + filename + properties.dataset.extension.img))
//    {
//        std::cerr << "Couldn't read color image \""
//                  << properties.dataset.path.img + filename + properties.dataset.extension.img << "\"." << std::endl;
//        return false;
//    }
//    if (!spRGB.read(properties.job.outline))
//    {
//        std::cerr << "Couldn't read superpixel image \"" << properties.job.outline << "\"." << std::endl;
//        return false;
//    }
//    auto cmap = helper::image::generateColorMap(properties.dataset.constants.numClusters);
//    LabelImage sp = helper::image::decolorize(spRGB, cmap);
//
//    // Compute outline
//    RGBImage outlined = helper::image::outline(sp, color, properties.border);
//
//    cv::Mat outlinedMat = static_cast<cv::Mat>(outlined);
//    return cv::imwrite(properties.out + filename + properties.dataset.extension.img, outlinedMat);
    return false;
}

bool rescale(UtilProperties const& properties)
{
    // Read in file names
    std::vector<std::string> list = readLines(properties.job.rescale);
    auto cmap = helper::image::generateColorMapVOC(256);

    for (std::string const& file : list)
    {
        std::string filenameGt = file + properties.dataset.extension.gt;
        std::string pathGt = properties.dataset.path.gt + filenameGt;
        std::string outPathGt = properties.out + "gt/";

        // Ground truth image
        std::cout << outPathGt << filenameGt;
        RGBImage gt;
        if (!gt.read(pathGt))
        {
            std::cerr << " Couldn't read ground truth image \"" << pathGt << "\"." << std::endl;
            return false;
        }
        gt.rescale(properties.rescaleFactor, false);
        gt.write(outPathGt + filenameGt);
        std::cout << "\tOK" << std::endl;
    }

    return true;
}

bool match_gt(UtilProperties const& properties)
{
    // Read in file names
    std::vector<std::string> list = readLines(properties.job.matchGt);
    auto cmap = helper::image::generateColorMapVOC(256);

    for (std::string const& file : list)
    {
        std::string filenameFeat = file + properties.dataset.extension.img;
        std::string pathFeat = properties.dataset.path.img + filenameFeat;
        std::string filenameGt = file + properties.dataset.extension.gt;
        std::string pathGt = properties.dataset.path.gt + filenameGt;
        std::string outPathGt = properties.out + "gt/";

        std::cout << outPathGt << filenameGt;

        FeatureImage features;
        if(!features.read(pathFeat))
        {
            std::cout << "\tERROR" << std::endl;
            std::cerr << "Unable to read features from \"" << pathFeat << "\"" << std::endl;
            return false;
        }

        // Ground truth image
        RGBImage gt_rgb;
        if (!gt_rgb.read(pathGt))
        {
            std::cout << "\tERROR" << std::endl;
            std::cerr << " Couldn't read ground truth image \"" << pathGt << "\"." << std::endl;
            return false;
        }
        gt_rgb.rescale(features.width(), features.height(), false);
        LabelImage gt = helper::image::decolorize(gt_rgb, cmap);

        // Note: rescaling the label image doesn't work properly. Apparently opencv has issues rescaling one-channel images?

        auto errCode = helper::image::writePalettePNG(outPathGt + filenameGt, gt, cmap);
        if(errCode != helper::image::PNGError::Okay)
        {
            std::cout << "\tERROR" << std::endl;
            std::cerr << " Couldn't write rescaled ground truth image \"" << outPathGt + filenameGt << "\". Error Code: " << (int) errCode << std::endl;
            return false;
        }
        std::cout << "\tOK" << std::endl;
    }
    return true;
}

bool scale_up(UtilProperties const& properties)
{
    // Read in file names
    std::vector<std::string> list = readLines(properties.job.scaleUp);
    auto cmap = helper::image::generateColorMapVOC(256);

    for (std::string const& file : list)
    {
        std::string filenameRgb = file + properties.dataset.extension.rgb;
        std::string pathRgb = properties.dataset.path.rgb + filenameRgb;
        std::string filenameLabeling = file + properties.dataset.extension.gt;
        std::string pathLabeling = properties.in + "labeling/" + filenameLabeling;
        std::string filenameClustering = file + properties.dataset.extension.gt;
        std::string pathClustering = properties.in + "clustering/" + filenameClustering;
        std::string outPathLabeling = properties.out + "labeling/";
        std::string outPathClustering = properties.out + "clustering/";

        std::cout << filenameRgb;

        // Ground truth image
        RGBImage rgb;
        if (!rgb.read(pathRgb))
        {
            std::cout << "\tERROR" << std::endl;
            std::cerr << " Couldn't read rgb image \"" << pathRgb << "\"." << std::endl;
            return false;
        }

        // Labeling
        LabelImage labeling;
        auto ok = helper::image::readPalettePNG(pathLabeling, labeling, nullptr);
        if (ok != helper::image::PNGError::Okay)
        {
            std::cout << "\tERROR" << std::endl;
            std::cerr << " Couldn't read labeling \"" << pathLabeling << "\". Error Code: " << (int) ok << std::endl;
            return false;
        }

        // Clustering
        LabelImage clustering;
        ok = helper::image::readPalettePNG(pathClustering, clustering, nullptr);
        if (ok != helper::image::PNGError::Okay)
        {
            std::cout << "\tERROR" << std::endl;
            std::cerr << " Couldn't read clustering \"" << pathClustering << "\". Error Code: " << (int) ok << std::endl;
            return false;
        }

        // Make up crude "marginals"
        cv::Mat labelingMarginals(labeling.height(), labeling.width(), CV_32FC(properties.dataset.constants.numClasses), cv::Scalar(0));
        cv::Mat clusteringMarginals(clustering.height(), clustering.width(), CV_32FC(properties.param.numClusters), cv::Scalar(0));

        size_t const cols = labelingMarginals.cols;
        size_t const rows = labelingMarginals.rows;
        size_t const ch_lab = labelingMarginals.channels();
        size_t const ch_clu = clusteringMarginals.channels();

        for(int x = 0; x < labelingMarginals.cols; ++x)
        {
            for(int y = 0; y < labelingMarginals.rows; ++y)
            {
                ((float*)labelingMarginals.data)[cols * y * ch_lab + x * ch_lab + labeling.at(x, y)] = 1;
                ((float*)clusteringMarginals.data)[cols * y * ch_clu + x * ch_clu + clustering.at(x, y)] = 1;
            }
        }

        // Rescale
        labeling = LabelImage(rgb.width(), rgb.height());
        clustering = LabelImage(rgb.width(), rgb.height());
        cv::Mat labelingMarginalsResized, clusteringMarginalsResized;
        cv::resize(labelingMarginals, labelingMarginalsResized, cv::Size(rgb.width(), rgb.height()));
        cv::resize(clusteringMarginals, clusteringMarginalsResized, cv::Size(rgb.width(), rgb.height()));

        labelingMarginals = labelingMarginalsResized;
        clusteringMarginals = clusteringMarginalsResized;

        // Copy arg max back
        for(int x = 0; x < labelingMarginals.cols; ++x)
        {
            for(int y = 0; y < labelingMarginals.rows; ++y)
            {
                // Labeling
                float curMax = ((float*)labelingMarginals.data)[labelingMarginals.cols * y * ch_lab + x * ch_lab+ 0];
                Label curLabel = 0;
                for(int c = 1; c < labelingMarginals.channels(); ++c)
                {
                    float val = ((float*)labelingMarginals.data)[labelingMarginals.cols * y * ch_lab + x * ch_lab + c];
                    if(val > curMax)
                    {
                        curMax = val;
                        curLabel = c;
                    }
                }
                labeling.at(x, y) = curLabel;

                // Clustering
                curMax = ((float*)clusteringMarginals.data)[clusteringMarginals.cols * y * ch_clu + x * ch_clu + 0];
                curLabel = 0;
                for(int c = 1; c < clusteringMarginals.channels(); ++c)
                {
                    float val = ((float*)clusteringMarginals.data)[clusteringMarginals.cols * y * ch_clu + x * ch_clu + c];
                    if(val > curMax)
                    {
                        curMax = val;
                        curLabel = c;
                    }
                }
                clustering.at(x, y) = curLabel;
            }
        }

        // Write results to disk
        ok = helper::image::writePalettePNG(outPathLabeling + filenameLabeling, labeling, cmap);
        if(ok != helper::image::PNGError::Okay)
        {
            std::cout << "\tERROR" << std::endl;
            std::cerr << " Couldn't write rescaled labeling \"" << outPathLabeling + filenameLabeling << "\". Error Code: " << (int) ok << std::endl;
            return false;
        }
        ok = helper::image::writePalettePNG(outPathClustering + filenameClustering, clustering, cmap);
        if(ok != helper::image::PNGError::Okay)
        {
            std::cout << "\tERROR" << std::endl;
            std::cerr << " Couldn't write rescaled clustering \"" << outPathClustering + filenameClustering << "\". Error Code: " << (int) ok << std::endl;
            return false;
        }
        std::cout << "\tOK" << std::endl;
    }
    return true;
}

bool copyFixPNG(UtilProperties const& properties)
{
    boost::filesystem::path inPath = properties.job.copyFixPNG;
    boost::filesystem::path outPath = properties.out;

    if(!boost::filesystem::is_directory(inPath))
        return false;

    auto cmap = helper::image::generateColorMapVOC(256);

    for(auto iter : boost::filesystem::directory_iterator(inPath))
    {
        boost::filesystem::path file = iter;
        if(boost::filesystem::is_regular_file(file) && file.extension() == ".png")
        {
            std::string filename = file.stem().string();
            RGBImage inRGB;
            inRGB.read(file.string());

            LabelImage labeling = helper::image::decolorize(inRGB, cmap);

            std::string outFile = (outPath / (filename + ".png")).string();
            auto result = helper::image::writePalettePNG(outFile, labeling, cmap);
            if(result != helper::image::PNGError::Okay)
                std::cerr << "Couldn't write \"" << outFile << "\". Error code: " << (int) result << std::endl;
            else
                std::cout << "File \"" << outFile << "\" successfully written." << std::endl;

        }
    }

    return true;
}

int main(int argc, char** argv)
{
    UtilProperties properties;
    properties.read("properties/hseg_util.info");
    properties.fromCmd(argc, argv);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    if (!properties.job.showWeightFile.empty())
        showWeight(properties.job.showWeightFile, properties);

    if (!properties.job.writeWeightFileText.empty())
        writeWeightFileText(properties);

    if (!properties.job.fillGroundTruth.empty())
        fillGroundTruth(properties);

    if (!properties.job.pairwiseStatistics.empty())
        pairwiseStatistics(properties);

    if (!properties.job.maxLoss.empty())
        computeMaxLoss(properties);

    if (!properties.job.outline.empty())
        outline(properties);

    if (!properties.job.rescale.empty())
        rescale(properties);

    if (!properties.job.scaleUp.empty())
        scale_up(properties);

    if (!properties.job.matchGt.empty())
        match_gt(properties);

    if(!properties.job.copyFixPNG.empty())
        copyFixPNG(properties);

    return 0;
}