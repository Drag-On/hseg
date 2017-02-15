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
                               PROP_DEFINE_A(std::string, matchGt, "", --match_gt)
                               PROP_DEFINE_A(std::string, copyFixPNG, "", --fix_PNG)
                               PROP_DEFINE_A(std::string, prepareFeatTrain, "", --prepareFeatTrain)
                  )
                  GROUP_DEFINE(dataset,
                               PROP_DEFINE_A(std::string, list, "", -l)
                               GROUP_DEFINE(path,
                                            PROP_DEFINE_A(std::string, rgb, "", --rgb)
                                            PROP_DEFINE_A(std::string, img, "", --img)
                                            PROP_DEFINE_A(std::string, gt, "", --gt)
                               )
                               GROUP_DEFINE(extension,
                                            PROP_DEFINE_A(std::string, rgb, ".jpg", --rgb_ext)
                                            PROP_DEFINE_A(std::string, img, ".mat", --img_ext)
                                            PROP_DEFINE_A(std::string, gt, ".png", --gt_ext)
                               )
                               GROUP_DEFINE(constants,
                                            PROP_DEFINE_A(uint32_t, numClasses, 21, --numClasses)
                                            PROP_DEFINE_A(uint32_t, featDim, 512, --featDim)
                               )
                  )
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

bool prepareFeatTrain(UtilProperties const& properties)
{
    // Read in file names
    std::vector<std::string> list = readLines(properties.job.prepareFeatTrain);
    auto cmap = helper::image::generateColorMapVOC(256);

    for (std::string const& file : list)
    {
        std::cout << " > " << file << ": " << std::flush;

        std::string filenameRgb = file + properties.dataset.extension.rgb;
        std::string pathRgb = properties.dataset.path.rgb + filenameRgb;
        std::string filenameGt = file + properties.dataset.extension.gt;
        std::string pathGt = properties.dataset.path.gt + filenameGt;
        std::string outPathRgb = properties.out + "rgb/";
        std::string outPathGt = properties.out + "gt/";

        // Load an image
        RGBImage rgb;
        if(!rgb.read(pathRgb))
        {
            std::cerr << "Unable to load image \"" << pathRgb << "\"." << std::endl;
            return false;
        }
        cv::Mat rgb_cv = static_cast<cv::Mat>(rgb);
//        rgb_cv.convertTo(rgb_cv, CV_32FC3);

        // Load ground truth
        LabelImage gt;
        helper::image::PNGError err = helper::image::readPalettePNG(pathGt, gt, nullptr);
        if(err != helper::image::PNGError::Okay)
        {
            std::cerr << "Unable to load ground truth \"" << pathGt << "\". Error Code: " << (int) err << std::endl;
            return false;
        }
        cv::Mat gt_cv = static_cast<cv::Mat>(gt);
//        gt_cv.convertTo(gt_cv, CV_32FC1);

        // Scale to base size
        int const base_size = 512;
        int const long_side = base_size + 1;
        int new_rows = long_side;
        int new_cols = long_side;
        if(rgb_cv.rows > rgb_cv.cols)
            new_cols = static_cast<int>(std::round(long_side / (float)rgb_cv.rows * rgb_cv.cols));
        else
            new_rows = static_cast<int>(std::round(long_side / (float)rgb_cv.cols * rgb_cv.rows));
        cv::Mat rgb_resized, gt_resized;
        cv::resize(rgb_cv, rgb_resized, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
        cv::resize(gt_cv, gt_resized, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_NEAREST);

        // Crop out parts that have the right dimensions
        float const stride_rate = 2.f / 3.f;
        int const crop_size = 473;
        int const stride = static_cast<int>(std::ceil(crop_size * stride_rate));
        for(int y = 0; y <= rgb_resized.rows; y += stride)
        {
            int s_y = y;
            bool breakOnEnd = false;
            if(y + crop_size > rgb_resized.rows)
            {
                s_y = std::max(0, rgb_resized.rows - crop_size);
                breakOnEnd = true;
            }
            for(int x = 0; x <= rgb_resized.cols; x += stride)
            {
                int s_x = x;
                bool breakOnEnd = false;
                if(x + crop_size > rgb_resized.cols)
                {
                    s_x = std::max(0, rgb_resized.cols - crop_size);
                    breakOnEnd = true;
                }


                //
                // RGB image
                //

                // Crop
                int patchW = crop_size, patchH = crop_size;
                if(s_x + patchW > rgb_resized.cols)
                    patchW = rgb_resized.cols - s_x;
                if(y + patchH > rgb_resized.rows)
                    patchH = rgb_resized.rows - s_y;
                cv::Mat patch = rgb_resized(cv::Rect(s_x, s_y, patchW, patchH));

                // Pad with zeros
                int const pad_w = crop_size - patchW;
                int const pad_h = crop_size - patchH;
                cv::Mat padded_img(crop_size, crop_size, patch.type());
                cv::copyMakeBorder(patch, padded_img, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(0));

                //
                // Ground Truth
                //

                // Crop
                cv::Mat patch_gt = gt_resized(cv::Rect(s_x, s_y, patchW, patchH));

                // Pad with 255
                cv::Mat padded_gt(crop_size, crop_size, patch_gt.type());
                cv::copyMakeBorder(patch_gt, padded_gt, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(255));

                //
                // Also create flipped version of the image
                //

                cv::Mat padded_img_flip, padded_gt_flip;
                flip(padded_img, padded_img_flip, 1);
                flip(padded_gt, padded_gt_flip, 1);

                //
                // Write to file
                //
                std::string cropFileName = file + "_" + std::to_string(x / stride) + "_" + std::to_string(y / stride);
                std::string cropFileNameFlip = file + "_FLIP_" + std::to_string(x / stride) + "_" + std::to_string(y / stride);

                // RGB
                std::string rgbOut = outPathRgb + cropFileName + properties.dataset.extension.rgb;
                if(!cv::imwrite(rgbOut, padded_img))
                {
                    std::cerr << "Couldn't write RGB crop to \"" << rgbOut << "\"" << std::endl;
                    return false;
                }
                std::string rgbOutFlip = outPathRgb + cropFileNameFlip + properties.dataset.extension.rgb;
                if(!cv::imwrite(rgbOutFlip, padded_img_flip))
                {
                    std::cerr << "Couldn't write flipped RGB crop to \"" << rgbOutFlip << "\"" << std::endl;
                    return false;
                }

                // GT
                std::string gtOut = outPathGt + cropFileName + properties.dataset.extension.gt;
                auto err = helper::image::writePalettePNG(gtOut, padded_gt, cmap);
                if(err != helper::image::PNGError::Okay)
                {
                    std::cerr << "Couldn't write GT crop to \"" << gtOut << "\". Error Code: " << (int) err << std::endl;
                    return false;
                }
                std::string gtOutFlip = outPathGt + cropFileNameFlip + properties.dataset.extension.gt;
                err = helper::image::writePalettePNG(gtOutFlip, padded_gt_flip, cmap);
                if(err != helper::image::PNGError::Okay)
                {
                    std::cerr << "Couldn't write flipped GT crop to \"" << gtOutFlip << "\". Error Code: " << (int) err << std::endl;
                    return false;
                }

                if(breakOnEnd)
                    break;
            }
            if(breakOnEnd)
                break;
        }
        std::cout << "OK!" << std::endl;
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

    if (!properties.job.matchGt.empty())
        match_gt(properties);

    if(!properties.job.copyFixPNG.empty())
        copyFixPNG(properties);

    if(!properties.job.prepareFeatTrain.empty())
        prepareFeatTrain(properties);

    return 0;
}