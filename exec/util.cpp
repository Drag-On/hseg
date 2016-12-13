//
// Created by jan on 23.09.16.
//

#include <BaseProperties.h>
#include <Energy/WeightsVec.h>
#include <helper/image_helper.h>
#include <helper/coordinate_helper.h>
#include <boost/filesystem/path.hpp>
#include <Inference/k-prototypes/Clusterer.h>
#include <Energy/feature_weights.h>
#include <Energy/LossAugmentedEnergyFunction.h>

PROPERTIES_DEFINE(Util,
                  GROUP_DEFINE(job,
                               PROP_DEFINE_A(std::string, showWeightFile, "", -sw)
                               PROP_DEFINE_A(std::string, writeWeightFile, "", -ww)
                               PROP_DEFINE_A(std::string, writeWeightFileText, "", -wtw)
                               PROP_DEFINE_A(std::string, fillGroundTruth, "", -f)
                               PROP_DEFINE_A(std::string, estimatePairwiseSigmaSq, "", -ep)
                               PROP_DEFINE_A(std::string, estimateSpDistance, "", -ed)
                               PROP_DEFINE_A(std::string, pairwiseStatistics, "", -ps)
                               PROP_DEFINE_A(std::string, maxLoss, "", -ml)
                               PROP_DEFINE_A(std::string, outline, "", -ol)
                               PROP_DEFINE_A(std::string, rescale, "", -rs)
                  )
                  GROUP_DEFINE(Constants,
                               PROP_DEFINE(size_t, numClasses, 21u)
                               PROP_DEFINE(size_t, numClusters, 300u)
                               PROP_DEFINE(float, rescaleFactor, 0.5f)
                  )
                  GROUP_DEFINE(Paths,
                               PROP_DEFINE(std::string, out, "")
                               PROP_DEFINE(std::string, image, "")
                               PROP_DEFINE(std::string, groundTruth, "")
                               PROP_DEFINE(std::string, groundTruthSp, "")
                               PROP_DEFINE(std::string, unary, "")
                  )
                  GROUP_DEFINE(FileExtensions,
                               PROP_DEFINE(std::string, image, ".jpg")
                               PROP_DEFINE(std::string, groundTruth, ".png")
                               PROP_DEFINE(std::string, groundTruthSp, ".png")
                               PROP_DEFINE(std::string, unary, "_prob.dat")
                  )
                  GROUP_DEFINE(Colors,
                               PROP_DEFINE(ARG(std::array<unsigned short, 3>), border, ARG(std::array<unsigned short, 3>{255, 255, 255}))
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
    std::cout << "Feature weight: ";
    float f = 0;
    std::cin >> f;
    std::cout << "Class weight: ";
    float l = 0;
    std::cin >> l;
    std::cout << "==========" << std::endl;
    WeightsVec w(21ul, u, p, f, l);
    return w.write(weightFile);
}

bool writeWeightFileText(UtilProperties const& properties)
{
    size_t const numClasses = properties.Constants.numClasses;
    WeightsVec weightsVec(numClasses);
    if (!weightsVec.read(properties.job.writeWeightFileText))
    {
        std::cerr << "Couldn't read weight file \"" << properties.job.writeWeightFileText << "\"" << std::endl;
        return false;
    }
    WeightsVec const& w = weightsVec;

    std::ofstream unaryOut(properties.Paths.out + "weights.unary.csv");
    if (unaryOut.is_open())
    {
        unaryOut << w.unary(0);
        for (size_t i = 1; i < numClasses; ++i)
            unaryOut << "\t" << w.unary(i);
        unaryOut << std::endl;
        unaryOut.close();
    }
    else
        std::cerr << "Couldn't write unary weights to \"" << properties.Paths.out << "weights.unary.csv" << "\""
                  << std::endl;

    std::ofstream pairwiseOut(properties.Paths.out + "weights.pairwise.csv");
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
        std::cerr << "Couldn't write pairwise weights to \"" << properties.Paths.out << "weights.pairwise.csv" << "\""
                  << std::endl;

    std::ofstream spFeatureOut(properties.Paths.out + "weights.feature.csv");
    if (spFeatureOut.is_open())
    {
        spFeatureOut << w.feature() << std::endl;
        spFeatureOut.close();
    }
    else
        std::cerr << "Couldn't write feature weights to \"" << properties.Paths.out << "weights.feature.csv" << "\""
                  << std::endl;

    std::ofstream spClassOut(properties.Paths.out + "weights.sp.csv");
    if (spClassOut.is_open())
    {
        for (Label l1 = 0; l1 < numClasses; ++l1)
        {
            spClassOut << w.classWeight(l1, 0);
            for (Label l2 = 1; l2 < numClasses; ++l2)
                spClassOut << "\t" << w.classWeight(l1, l2);
            spClassOut << std::endl;
        }
        spClassOut.close();
    }
    else
        std::cerr << "Couldn't write superpixel weights to \"" << properties.Paths.out << "weights.sp.csv" << "\""
                  << std::endl;

    return true;
}

bool fillGroundTruth(UtilProperties const& properties)
{
    RGBImage gtRGB;
    if (!gtRGB.read(properties.job.fillGroundTruth))
    {
        std::cerr << "Couldn't open ground truth image from \"" << properties.job.fillGroundTruth << "\"." << std::endl;
        return false;
    }
    auto cmap = helper::image::generateColorMapVOC(256);
    LabelImage gt = helper::image::decolorize(gtRGB, cmap);
    LabelImage fixedGt = gt;

    // Find invalid pixels
    size_t const numClasses = properties.Constants.numClasses;
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
    RGBImage result = helper::image::colorize(fixedGt, cmap);
    cv::Mat resultMat = static_cast<cv::Mat>(result);
    std::string filename = boost::filesystem::path(properties.job.fillGroundTruth).filename().string();
    if (!cv::imwrite(properties.Paths.out + filename, resultMat))
    {
        std::cerr << "Couldn't write result to \"" << properties.Paths.out + filename << "\"." << std::endl;
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

bool estimatePairwiseSigmaSq(UtilProperties const& properties)
{
    // Read in file names
    std::vector<std::string> list = readLines(properties.job.estimatePairwiseSigmaSq);

    auto cmap = helper::image::generateColorMapVOC(256);

    std::vector<double> means;
    std::vector<double> variances;

    // Read in files from the specified folder
    for (auto const& file : list)
    {
        // Read in color image and ground truth image
        RGBImage color, gtRGB;
        std::string clrFileName = properties.Paths.image + file + properties.FileExtensions.image;
        if (!color.read(clrFileName))
        {
            std::cerr << "Couldn't read color image \"" << clrFileName << "\"." << std::endl;
            return false;
        }
        std::string gtFileName = properties.Paths.groundTruth + file + properties.FileExtensions.groundTruth;
        if (!gtRGB.read(gtFileName))
        {
            std::cerr << "Couldn't read ground truth image \"" << gtFileName << "\"." << std::endl;
            return false;
        }
        CieLabImage cielab = color.getCieLabImg();
        LabelImage gt = helper::image::decolorize(gtRGB, cmap);

        // Compute samples
        std::vector<double> data;
        size_t N = 0;
        for (size_t i = 0; i < cielab.pixels(); ++i)
        {
            auto coords = helper::coord::siteTo2DCoordinate(i, cielab.width());
            Label l = gt.atSite(i);
            if (coords.x() < cielab.width() - 1)
            {
                Label lr = gt.at(coords.x() + 1, coords.y());
                if (l != lr)
                {
                    N++;
                    double point = std::pow(cielab.atSite(i, 0) - cielab.at(coords.x() + 1, coords.y(), 0), 2) +
                                   std::pow(cielab.atSite(i, 1) - cielab.at(coords.x() + 1, coords.y(), 1), 2) +
                                   std::pow(cielab.atSite(i, 2) - cielab.at(coords.x() + 1, coords.y(), 2), 2);
                    data.push_back(point);
                }
            }
            if (coords.y() < cielab.height() - 1)
            {
                Label ld = gt.at(coords.x(), coords.y() + 1);
                if (l != ld)
                {
                    N++;
                    double point = std::pow(cielab.atSite(i, 0) - cielab.at(coords.x(), coords.y() + 1, 0), 2) +
                                   std::pow(cielab.atSite(i, 1) - cielab.at(coords.x(), coords.y() + 1, 1), 2) +
                                   std::pow(cielab.atSite(i, 2) - cielab.at(coords.x(), coords.y() + 1, 2), 2);
                    data.push_back(point);
                }
            }
        }

        assert(N > 0);

        // Estimate ML mean for this image
        double mean = 0;
        for (auto const& d : data)
            mean += d;
        mean /= N;

        assert(!std::isnan(mean));

        // Estimate ML variance for this image
        double variance = 0;
        for (auto const& d : data)
            variance += std::pow(d - mean, 2);
        variance /= N;

        // Store results
        means.push_back(mean);
        variances.push_back(variance);
    }

    // Compute mean mean over all images
    double meanMean = std::accumulate(means.begin(), means.end(), 0.);
    meanMean /= means.size();
    std::cout << "Mean sample mean: " << meanMean << std::endl;

    // Compute mean variance over all images
    double meanVariance = std::accumulate(variances.begin(), variances.end(), 0.);
    meanVariance /= variances.size();
    std::cout << "Mean sample variance: " << meanVariance << std::endl;

    // Compute actual sigma square value
    double sigmaSq = 1 / (2 * meanVariance);
    std::cout << "Pairwise sigma square should be " << sigmaSq << std::endl;

    return true;
}

bool estimateSpDistance(UtilProperties const& properties)
{
    // Read in files to consider
    std::vector<std::string> list = readLines(properties.job.estimateSpDistance);

    auto cmap = helper::image::generateColorMapVOC(256);
    auto cmap2 = helper::image::generateColorMap(properties.Constants.numClusters);

    UnaryFile fakeUnary;
    WeightsVec fakeWeights(properties.Constants.numClasses);
    Matrix5f fakeFeatureWeights;
    EnergyFunction energy(fakeUnary, fakeWeights, 0, fakeFeatureWeights);

    // Compute mean
    Vector5f mean = Vector5f::Zero();
    size_t N = 0;
    for (auto const& s : list)
    {
        std::string imageFile = properties.Paths.image + s + properties.FileExtensions.image;
        std::string gtFile = properties.Paths.groundTruth + s + properties.FileExtensions.groundTruth;
        std::string gtSpFile = properties.Paths.groundTruthSp + s + properties.FileExtensions.groundTruthSp;
        RGBImage image, gtRGB, gtSpRGB;
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
        if (!gtSpRGB.read(gtSpFile))
        {
            std::cerr << "Couldn't read ground truth superpixel image \"" << gtSpFile << "\"." << std::endl;
            return false;
        }
        LabelImage gtSp = helper::image::decolorize(gtSpRGB, cmap2);

        CieLabImage cielab = image.getCieLabImg();
        auto clusters = Clusterer<EnergyFunction>::computeClusters(gtSp, cielab, gt, properties.Constants.numClusters,
                                                                   properties.Constants.numClasses, energy);

        for (size_t i = 0; i < gtSp.pixels(); ++i)
        {
            Label l = gtSp.atSite(i);
            auto coords = helper::coord::siteTo2DCoordinate(i, cielab.width());

            Vector5f vec;
            vec(0) = cielab.atSite(i, 0) - clusters[l].mean.r();
            vec(1) = cielab.atSite(i, 1) - clusters[l].mean.g();
            vec(2) = cielab.atSite(i, 2) - clusters[l].mean.b();
            vec(3) = coords.x() - clusters[l].mean.x();
            vec(4) = coords.y() - clusters[l].mean.y();

            mean += vec;
        }
        N += gtSp.pixels();
    }
    mean /= N;

    // Estimate sample covariance
    Matrix5f cov = Matrix5f::Zero();
    for (auto const& s : list)
    {
        std::string imageFile = properties.Paths.image + s + properties.FileExtensions.image;
        std::string gtFile = properties.Paths.groundTruth + s + properties.FileExtensions.groundTruth;
        std::string gtSpFile = properties.Paths.groundTruthSp + s + properties.FileExtensions.groundTruthSp;
        RGBImage image, gtRGB, gtSpRGB;
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
        if (!gtSpRGB.read(gtSpFile))
        {
            std::cerr << "Couldn't read ground truth superpixel image \"" << gtSpFile << "\"." << std::endl;
            return false;
        }
        LabelImage gtSp = helper::image::decolorize(gtSpRGB, cmap2);

        CieLabImage cielab = image.getCieLabImg();
        auto clusters = Clusterer<EnergyFunction>::computeClusters(gtSp, cielab, gt, properties.Constants.numClusters,
                                                                   properties.Constants.numClasses, energy);

        // Compute sum
        for (size_t i = 0; i < gtSp.pixels(); ++i)
        {
            Label l = gtSp.atSite(i);
            auto coords = helper::coord::siteTo2DCoordinate(i, cielab.width());

            Vector5f vec;
            vec(0) = cielab.atSite(i, 0) - clusters[l].mean.r();
            vec(1) = cielab.atSite(i, 1) - clusters[l].mean.g();
            vec(2) = cielab.atSite(i, 2) - clusters[l].mean.b();
            vec(3) = coords.x() - clusters[l].mean.x();
            vec(4) = coords.y() - clusters[l].mean.y();

            cov += (vec - mean) * (vec - mean).transpose();
        }
    }
    cov /= N - 1;

    // Show results
    std::cout << cov << std::endl;

    // Write to file
    writeFeatureWeights(properties.Paths.out + "featureWeights.txt", cov);

    return true;
}

bool pairwiseStatistics(UtilProperties const& properties)
{
    // Read in files to consider
    std::vector<std::string> list = readLines(properties.job.pairwiseStatistics);

    auto cmap = helper::image::generateColorMapVOC(256);

    size_t pairwiseConnections = 0;
    std::vector<float> pairwiseWeights((properties.Constants.numClasses * properties.Constants.numClasses) / 2, 0.f);

    // Iterate them
    for (auto const& s : list)
    {
        std::string imageFile = properties.Paths.image + s + properties.FileExtensions.image;
        std::string gtFile = properties.Paths.groundTruth + s + properties.FileExtensions.groundTruth;
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
            decltype(coords) coordsR = {coords.x() + 1, coords.y()};
            decltype(coords) coordsD = {coords.x(), coords.y() + 1};
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

    for (size_t l1 = 0; l1 < properties.Constants.numClasses; ++l1)
    {
        for (size_t l2 = 0; l2 < properties.Constants.numClasses; ++l2)
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
        std::string imageFile = properties.Paths.image + s + properties.FileExtensions.image;
        std::string gtFile = properties.Paths.groundTruth + s + properties.FileExtensions.groundTruth;
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
        float lossFactor = LossAugmentedEnergyFunction::computeLossFactor(gt, properties.Constants.numClasses);

        maxLoss += image.pixels() * lossFactor;
    }
    maxLoss /= list.size();

    std::cout << "Maximum loss from set \"" << properties.job.maxLoss << "\" (" << list.size() << " images): "
              << maxLoss << std::endl;
    return true;
}

bool outline(UtilProperties const& properties)
{
    // Load color and superpixel image
    std::string filename = boost::filesystem::path(properties.job.outline).stem().string();
    RGBImage color, spRGB;
    if (!color.read(properties.Paths.image + filename + properties.FileExtensions.image))
    {
        std::cerr << "Couldn't read color image \""
                  << properties.Paths.image + filename + properties.FileExtensions.image << "\"." << std::endl;
        return false;
    }
    if (!spRGB.read(properties.job.outline))
    {
        std::cerr << "Couldn't read superpixel image \"" << properties.job.outline << "\"." << std::endl;
        return false;
    }
    auto cmap = helper::image::generateColorMap(properties.Constants.numClusters);
    LabelImage sp = helper::image::decolorize(spRGB, cmap);

    // Compute outline
    RGBImage outlined = helper::image::outline(sp, color, properties.Colors.border);

    cv::Mat outlinedMat = static_cast<cv::Mat>(outlined);
    return cv::imwrite(properties.Paths.out + filename + properties.FileExtensions.image, outlinedMat);
}

bool rescale(UtilProperties const& properties)
{
    // Read in file names
    std::vector<std::string> list = readLines(properties.job.rescale);
    auto cmap = helper::image::generateColorMapVOC(255);

    for (std::string const& file : list)
    {
        std::string filenameColor = file + properties.FileExtensions.image;
        std::string filenameGt = file + properties.FileExtensions.groundTruth;
        std::string filenameUnary = file + properties.FileExtensions.unary;
        std::string pathColor = properties.Paths.image + filenameColor;
        std::string pathGt = properties.Paths.groundTruth + filenameGt;
        std::string pathUnary = properties.Paths.unary + filenameUnary;
        std::string outPathColor = properties.Paths.out + "color/";
        std::string outPathGt = properties.Paths.out + "gt/";
        std::string outPathUnary = properties.Paths.out + "unaries/";

        // Color image
        std::cout << outPathColor << filenameColor;
        RGBImage img;
        if (!img.read(pathColor))
        {
            std::cerr << " Couldn't read image \"" << pathColor << "\"." << std::endl;
            return false;
        }
        img.rescale(properties.Constants.rescaleFactor, true);
        img.write(outPathColor + filenameColor);
        std::cout << "\tOK" << std::endl;

        // Ground truth image
        std::cout << outPathGt << filenameGt;
        RGBImage gt;
        if (!gt.read(pathGt))
        {
            std::cerr << " Couldn't read ground truth image \"" << pathGt << "\"." << std::endl;
            return false;
        }
        gt.rescale(properties.Constants.rescaleFactor, false);
        gt.write(outPathGt + filenameGt);
        std::cout << "\tOK" << std::endl;

        // Unary
        std::cout << outPathUnary << filenameUnary;
        UnaryFile unary;
        if (!unary.read(pathUnary))
        {
            std::cerr << " Couldn't read unary \"" << pathUnary << "\"." << std::endl;
            return false;
        }
        unary.rescale(properties.Constants.rescaleFactor);
        unary.write(outPathUnary + filenameUnary);
        auto maxLabeling = unary.maxLabeling();
        auto maxLabelingImg = helper::image::colorize(maxLabeling, cmap);
        maxLabelingImg.write(outPathUnary + filenameGt);
        std::cout << "\tOK" << std::endl;
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

    if (!properties.job.writeWeightFile.empty())
        writeWeight(properties.job.writeWeightFile);

    if (!properties.job.writeWeightFileText.empty())
        writeWeightFileText(properties);

    if (!properties.job.fillGroundTruth.empty())
        fillGroundTruth(properties);

    if (!properties.job.estimatePairwiseSigmaSq.empty())
        estimatePairwiseSigmaSq(properties);

    if (!properties.job.estimateSpDistance.empty())
        estimateSpDistance(properties);

    if (!properties.job.pairwiseStatistics.empty())
        pairwiseStatistics(properties);

    if (!properties.job.maxLoss.empty())
        computeMaxLoss(properties);

    if (!properties.job.outline.empty())
        outline(properties);

    if (!properties.job.rescale.empty())
        rescale(properties);

    return 0;
}