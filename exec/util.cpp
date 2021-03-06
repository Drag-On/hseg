//
// Created by jan on 23.09.16.
//

#include <BaseProperties.h>
#include <Energy/Weights.h>
#include <helper/image_helper.h>
#include <helper/coordinate_helper.h>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <Energy/LossAugmentedEnergyFunction.h>
#include <densecrf.h>
#include <helper/utility.h>
#include <Inference/InferenceIterator.h>

#ifdef WITH_CAFFE
#include <caffe/util/db.hpp>
#endif

PROPERTIES_DEFINE(Util,
                  GROUP_DEFINE(job,
                               PROP_DEFINE_A(std::string, showWeightFile, "", --show)
                               PROP_DEFINE_A(std::string, writeWeightFileText, "", --toText)
                               PROP_DEFINE_A(std::string, fillGroundTruth, "", --fill_gt)
                               PROP_DEFINE_A(std::string, pairwiseStatistics, "", --pair_stats)
                               PROP_DEFINE_A(std::string, maxLoss, "", --max_loss)
                               PROP_DEFINE_A(std::string, outline, "", --outline)
                               PROP_DEFINE_A(std::string, rescale, "", --rescale)
                               PROP_DEFINE_A(std::string, post_pro, "", --post_pro)
                               PROP_DEFINE_A(std::string, matchGt, "", --match_gt)
                               PROP_DEFINE_A(std::string, copyFixPNG, "", --fix_PNG)
                               PROP_DEFINE_A(std::string, prepareDataset, "", --prepareDataset)
                               PROP_DEFINE_A(std::string, writeLMDB, "", --writeLMDB)
                               PROP_DEFINE_A(std::string, createFakeMarginals, "", --createFakeMarginals)
                               PROP_DEFINE_A(std::string, stitchMarginals, "", --stitchMarginals)
                               PROP_DEFINE_A(std::string, createBasicFeatures, "", --createBasicFeatures)
                               PROP_DEFINE_A(std::string, mergeFeatures, "", --mergeFeatures)
                               PROP_DEFINE_A(std::string, testIterationProgress, "", --testIterationProgress)
                               PROP_DEFINE_A(std::string, symmetryCheck, "", --symmetryCheck)
                               PROP_DEFINE_A(std::string, prepCityscapesGt, "", --prepCityscapesGt)
                               PROP_DEFINE_A(std::string, figureGroundToPascal, "", --figureGroundToPascal)
                  )
                  GROUP_DEFINE(datasetPx,
                               PROP_DEFINE_A(std::string, list, "", -l)
                               GROUP_DEFINE(path,
                                            PROP_DEFINE_A(std::string, rgb, "", --rgb)
                                            PROP_DEFINE_A(std::string, img, "", --img)
                                            PROP_DEFINE_A(std::string, gt, "", --gt)
                                            PROP_DEFINE_A(std::string, rgb_orig, "", --rgb_orig)
                                            PROP_DEFINE_A(std::string, gt_orig, "", --gt_orig)
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
                  GROUP_DEFINE(datasetCluster,
                               GROUP_DEFINE(path,
                                            PROP_DEFINE_A(std::string, img, "", --img_clu)
                               )
                               GROUP_DEFINE(extension,
                                            PROP_DEFINE_A(std::string, img, ".mat", --img_clu_ext)
                               )
                               GROUP_DEFINE(constants,
                                            PROP_DEFINE_A(uint32_t, featDim, 512, --featDimClu)
                               )
                  )
                  GROUP_DEFINE(prepareDataset,
                          PROP_DEFINE_A(int, baseSize, 512, --base_size)
                          PROP_DEFINE_A(int, cropSize, 473, --crop_size)
                          PROP_DEFINE_A(bool, withGt, true, --with_gt)
                          PROP_DEFINE_A(bool, cityscapes, false, --cityscapes)
                  )
                  GROUP_DEFINE(mergeFeatures,
                          PROP_DEFINE_A(std::string, first, "", --first)
                          PROP_DEFINE_A(std::string, second, "", --second)
                  )
                  GROUP_DEFINE(figureGroundToPascal,
                          PROP_DEFINE_A(Label, groundLabel, 0, --ground)
                          PROP_DEFINE_A(Label, figureLabel, 1, --figure)
                          PROP_DEFINE_A(std::string, inExt, ".jpg", --inExt)
                          PROP_DEFINE_A(std::string, outExt, ".png", --outExt)
                          PROP_DEFINE_A(float, threshold, 128, --thresh)
                  )
                  GROUP_DEFINE(param,
                          PROP_DEFINE_A(ClusterId, numClusters, 100, --numClusters)
                          PROP_DEFINE_A(bool, usePairwise, false, --usePairwise)
                          PROP_DEFINE_A(float, eps, 0, --eps)
                          PROP_DEFINE_A(float, maxIter, 50, --max_iter)
                  )
                  PROP_DEFINE_A(std::string, in, "", -i)
                  PROP_DEFINE_A(std::string, out, "", -o)
                  PROP_DEFINE_A(float, rescaleFactor, 0.5f, --rescale)
                  PROP_DEFINE_A(ARG(std::array<unsigned short, 3>), border, ARG(std::array<unsigned short, 3>{255, 255, 255}), --color)
)

bool showWeight(std::string const& weightFile, UtilProperties const& properties)
{
    Weights w(properties.datasetPx.constants.numClasses, properties.datasetPx.constants.featDim, properties.datasetCluster.constants.featDim);
    if (!w.read(weightFile))
    {
        std::cerr << "Couldn't read weight file \"" << weightFile << "\"" << std::endl;
        return false;
    }
    std::cout << "==========" << std::endl;
//    std::cout << weightFile << ":" << std::endl;
//    std::cout << w << std::endl;
//    std::cout << std::endl;
    w.printStats();
    std::cout << "==========" << std::endl;
    return true;
}

bool writeWeightFileText(UtilProperties const& properties)
{
    size_t const numClasses = properties.datasetPx.constants.numClasses;
    size_t const featDim = properties.datasetPx.constants.featDim;
    size_t const featDimCluster = properties.datasetCluster.constants.featDim;
    Weights weightsVec(numClasses, featDim, featDimCluster);
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
    size_t const numClasses = properties.datasetPx.constants.numClasses;
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

bool computeMaxLoss(UtilProperties const& properties)
{
    // Read in files to consider
    std::vector<std::string> list = readLines(properties.job.maxLoss);

    auto cmap = helper::image::generateColorMapVOC(256);

    // Iterate them
    float maxLoss = 0.f;
    for (auto const& s : list)
    {
        std::string imageFile = properties.datasetPx.path.img + s + properties.datasetPx.extension.img;
        std::string gtFile = properties.datasetPx.path.gt + s + properties.datasetPx.extension.gt;
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
        float lossFactor = LossAugmentedEnergyFunction::computeLossFactor(gt, properties.datasetPx.constants.numClasses);

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
        std::string filenameGt = file + properties.datasetPx.extension.gt;
        std::string pathGt = properties.datasetPx.path.gt + filenameGt;
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
        std::string filenamePxFeat = file + properties.datasetPx.extension.img;
        std::string pathPxFeat = properties.datasetPx.path.img + filenamePxFeat;
        std::string filenameGt = file + properties.datasetPx.extension.gt;
        std::string pathGt = properties.datasetPx.path.gt + filenameGt;
        std::string outPathGt = properties.out + "gt/";

        std::cout << outPathGt << filenameGt;

        FeatureImage featuresPx;
        if(!featuresPx.read(pathPxFeat))
        {
            std::cout << "\tERROR" << std::endl;
            std::cerr << "Unable to read features from \"" << pathPxFeat << "\"" << std::endl;
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
        gt_rgb.rescale(featuresPx.width(), featuresPx.height(), false);
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

bool post_pro(UtilProperties const& properties)
{
    // Read in file names
    std::vector<std::string> list = readLines(properties.job.post_pro);
    auto cmap = helper::image::generateColorMapVOC(256);

    for (std::string const& file : list)
    {
        std::string filenameRgb = file + properties.datasetPx.extension.rgb;
        std::string pathRgb = properties.datasetPx.path.rgb_orig + filenameRgb;
        std::string filenameMarginals = file + ".mat";
        std::string pathMarginals = properties.in + filenameMarginals;
        std::string outFilenameLabeling = file + properties.datasetPx.extension.gt;
        std::string outPathLabeling = properties.out + outFilenameLabeling;

        std::cout << file;

        // Ground truth image
        RGBImage rgb;
        if (!rgb.read(pathRgb))
        {
            std::cout << "\tERROR" << std::endl;
            std::cerr << " Couldn't read rgb image \"" << pathRgb << "\"." << std::endl;
            return false;
        }

        // Marginals
        FeatureImage marginals;
        if(!marginals.read(pathMarginals))
        {
            std::cout << "\tERROR" << std::endl;
            std::cerr << " Couldn't read marginals \"" << pathMarginals << "\"." << std::endl;
            return false;
        }

        if(rgb.width() != marginals.width() || rgb.height() != marginals.height() || marginals.dim() != properties.datasetPx.constants.numClasses)
        {
            std::cout << "\tERROR" << std::endl;
            std::cerr << " Marginal size doesn't match original image." << std::endl;
            return false;
        }

        // Do Dense CRF inference
        // Store it in a way the dense crf implementation understands
        Label const numClasses = properties.datasetPx.constants.numClasses;
        Eigen::MatrixXf unary(numClasses, rgb.width() * rgb.height());
        for(Coord y = 0; y < rgb.height(); ++y)
        {
            for(Coord x = 0; x < rgb.width(); ++x)
            {
                for(Label l = 0; l < numClasses; ++l)
                    unary(l, x + y * rgb.width()) = -marginals.at(x, y)[l];
            }
        }

        // Store image in a way the dense crf implementation understands
        std::vector<unsigned char> im(rgb.width() * rgb.height() * 3, 0);
        for(Coord x = 0; x < rgb.width(); ++x)
        {
            for(Coord y = 0; y < rgb.height(); ++y)
            {
                for(uint32_t c = 0; c < 3; ++c)
                    im[c + x * 3 + y * 3 * rgb.width()] = rgb.at(x, y, c);
            }
        }

        // Setup the CRF model
        DenseCRF2D crf(rgb.width(), rgb.height(), numClasses);
        // Specify the unary potential as an array of size W*H*(#classes)
        // packing order: x0y0l0 x0y0l1 x0y0l2 .. x1y0l0 x1y0l1 ...
        crf.setUnaryEnergy( unary );
        // add a color independent term (feature = pixel location 0..W-1, 0..H-1)
        // x_stddev = 3
        // y_stddev = 3
        // weight = 3
        crf.addPairwiseGaussian( 3, 3, new PottsCompatibility( 1 ) );
        // add a color dependent term (feature = xyrgb)
        // x_stddev = 60
        // y_stddev = 60
        // r_stddev = g_stddev = b_stddev = 20
        // weight = 10
        crf.addPairwiseBilateral( 30, 30, 13, 13, 13, im.data(), new PottsCompatibility( 3 ) );

        // Do map inference
        VectorXs map = crf.map(5);

        // Copy result to label image
        LabelImage labeling(rgb.width(), rgb.height());
        for (Coord x = 0; x < rgb.width(); ++x)
        {
            for (Coord y = 0; y < rgb.height(); ++y)
                labeling.at(x, y) = map(x + y * rgb.width());
        }

        // Write results to disk
        auto ok = helper::image::writePalettePNG(outPathLabeling, labeling, cmap);
        if(ok != helper::image::PNGError::Okay)
        {
            std::cout << "\tERROR" << std::endl;
            std::cerr << " Couldn't write rescaled labeling \"" << outPathLabeling << "\". Error Code: " << (int) ok << std::endl;
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

bool prepareDataset(UtilProperties const& properties)
{
    // Read in file names
    std::vector<std::string> list = readLines(properties.job.prepareDataset);
    std::vector<std::string> infoList;
    auto cmap = helper::image::generateColorMapVOC(256);

    for (std::string const& file : list)
    {
        std::cout << " > " << file << ": " << std::flush;

        std::string filenameRgb = file + properties.datasetPx.extension.rgb;
        std::string pathRgb = properties.datasetPx.path.rgb + filenameRgb;
        std::string filenameGt = file + properties.datasetPx.extension.gt;
        std::string pathGt = properties.datasetPx.path.gt + filenameGt;
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
        rgb_cv.convertTo(rgb_cv, CV_8UC3);

        // Load ground truth
        LabelImage gt;
        cv::Mat gt_cv;
        if(properties.prepareDataset.withGt)
        {
            if(properties.prepareDataset.cityscapes)
            {
                if(!gt.read(pathGt))
                {
                    std::cerr << "Unable to load ground truth \"" << pathGt << "\"." << std::endl;
                    return false;
                }
                // Cityscapes has many invalid labels. The valid labels are not in a consecutive order, therefore we
                // need to fix the labels here.
                for(SiteId s = 0; s < gt.pixels(); ++s)
                {
                    Label const l = gt.atSite(s);
                    Label newL;
                    if((l >= 0 && l <= 6) || (l>= 9 && l <= 10) || (l >= 14 && l <= 16) || (l == 18) || (l >=29 && l <= 30))
                        newL = 255;
                    else if (l > 30)
                        newL = l - 15;
                    else if (l > 18)
                        newL = l - 13;
                    else if (l > 16)
                        newL = l - 12;
                    else if (l > 10)
                        newL = l - 9;
                    else if (l > 6)
                        newL = l - 7;
                    gt.atSite(s) = newL;
                }
            }
            else
            {
                helper::image::PNGError err = helper::image::readPalettePNG(pathGt, gt, nullptr);
                if(err != helper::image::PNGError::Okay)
                {
                    std::cerr << "Unable to load ground truth \"" << pathGt << "\". Error Code: " << (int) err << std::endl;
                    return false;
                }
            }
            gt_cv = static_cast<cv::Mat>(gt);
            gt_cv.convertTo(gt_cv, CV_8UC1);
        }

        // Scale to base size
        int const base_size = properties.prepareDataset.baseSize;
        int const long_side = base_size + 1;
        int new_rows = long_side;
        int new_cols = long_side;
        if(rgb_cv.rows > rgb_cv.cols)
            new_cols = static_cast<int>(std::round(long_side / (float)rgb_cv.rows * rgb_cv.cols));
        else
            new_rows = static_cast<int>(std::round(long_side / (float)rgb_cv.cols * rgb_cv.rows));
        cv::Mat rgb_resized, gt_resized;
        cv::resize(rgb_cv, rgb_resized, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
        if(properties.prepareDataset.withGt)
            cv::resize(gt_cv, gt_resized, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_NEAREST);

        // Crop out parts that have the right dimensions
        float const stride_rate = 2.f / 3.f;
        int const crop_size = properties.prepareDataset.cropSize;
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

                cv::Mat padded_gt;
                if(properties.prepareDataset.withGt)
                {
                    // Crop
                    cv::Mat patch_gt = gt_resized(cv::Rect(s_x, s_y, patchW, patchH));

                    // Pad with 255
                    padded_gt = cv::Mat(crop_size, crop_size, patch_gt.type());
                    cv::copyMakeBorder(patch_gt, padded_gt, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(255));
                }

                //
                // Also create flipped version of the image
                //

                cv::Mat padded_img_flip, padded_gt_flip;
                flip(padded_img, padded_img_flip, 1);
                if(properties.prepareDataset.withGt)
                    flip(padded_gt, padded_gt_flip, 1);

                //
                // Write to file
                //
                std::string cropFileName = file + "_" + std::to_string(x / stride) + "_" + std::to_string(y / stride);
                std::string cropFileNameFlip = file + "_FLIP_" + std::to_string(x / stride) + "_" + std::to_string(y / stride);

                // RGB
                std::string rgbOut = outPathRgb + cropFileName + properties.datasetPx.extension.rgb;
                if(!cv::imwrite(rgbOut, padded_img))
                {
                    std::cerr << "Couldn't write RGB crop to \"" << rgbOut << "\"" << std::endl;
                    return false;
                }
                std::string rgbOutFlip = outPathRgb + cropFileNameFlip + properties.datasetPx.extension.rgb;
                if(!cv::imwrite(rgbOutFlip, padded_img_flip))
                {
                    std::cerr << "Couldn't write flipped RGB crop to \"" << rgbOutFlip << "\"" << std::endl;
                    return false;
                }

                // GT
                if(properties.prepareDataset.withGt)
                {
                    std::string gtOut = outPathGt + cropFileName + properties.datasetPx.extension.gt;
                    auto err = helper::image::writePalettePNG(gtOut, padded_gt, cmap);
                    if(err != helper::image::PNGError::Okay)
                    {
                        std::cerr << "Couldn't write GT crop to \"" << gtOut << "\". Error Code: " << (int) err << std::endl;
                        return false;
                    }
                    std::string gtOutFlip = outPathGt + cropFileNameFlip + properties.datasetPx.extension.gt;
                    err = helper::image::writePalettePNG(gtOutFlip, padded_gt_flip, cmap);
                    if(err != helper::image::PNGError::Okay)
                    {
                        std::cerr << "Couldn't write flipped GT crop to \"" << gtOutFlip << "\". Error Code: " << (int) err << std::endl;
                        return false;
                    }
                }

                // Write additional info
                std::string info = cropFileName + ";" +
                                   std::to_string(s_x) + ";" +
                                   std::to_string(s_y) + ";" +
                                   std::to_string(patchW) + ";" +
                                   std::to_string(patchH) + ";" +
                                   file + ";" +
                                   "false";
                std::string infoFlip = cropFileNameFlip + ";" +
                                       std::to_string(s_x) + ";" +
                                       std::to_string(s_y) + ";" +
                                       std::to_string(patchW) + ";" +
                                       std::to_string(patchH) + ";" +
                                       file + ";" +
                                       "true";

                infoList.push_back(info);
                infoList.push_back(infoFlip);

                if(breakOnEnd)
                    break;
            }
            if(breakOnEnd)
                break;
        }
        std::cout << "OK!" << std::endl;
    }

    // Write list with images to file
    std::ofstream outMetadata(properties.out + "metadata.txt");
    if(outMetadata.is_open())
    {
        for (auto const& l : infoList)
            outMetadata << l << std::endl;
        outMetadata.close();
    }
    else
        std::cerr << "Unable to write meta data to \"" << properties.out + "metadata.txt" << "\"" << std::endl;

    return true;
}

bool writeLMDB(UtilProperties const& properties)
{
#ifdef WITH_CAFFE
    // Read in file names
    std::vector<std::string> listfile = readLines(properties.job.writeLMDB);

    std::cout << listfile.size() << " crops." << std::endl;

    // Shuffle indices
    std::vector<size_t> indices(listfile.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    auto cmap = helper::image::generateColorMapVOC(256);

    // Create database
    std::string dbFilename = properties.out + "database_lmdb";
    std::unique_ptr<caffe::db::DB> db(caffe::db::GetDB("lmdb"));
    db->Open(dbFilename, caffe::db::NEW);
    std::unique_ptr<caffe::db::Transaction> transaction(db->NewTransaction());

    for (size_t const& i : indices)
    {
        std::string filename = listfile[i];
        std::string filenameRgb = properties.dataset.path.rgb + filename + properties.dataset.extension.rgb;
        std::string filenameGt = properties.dataset.path.gt + filename + properties.dataset.extension.gt;

        std::cout << " > " << i << ": \"" << filename << "\"" << std::flush;

        // Load an image
        RGBImage rgb;
        if(!rgb.read(filenameRgb))
        {
            std::cerr << "Unable to load image \"" << filenameRgb << "\"." << std::endl;
            db->Close();
            return false;
        }
        cv::Mat rgb_cv = static_cast<cv::Mat>(rgb);
        rgb_cv.convertTo(rgb_cv, CV_8UC3);

        // Load ground truth
        LabelImage gt;
        helper::image::PNGError err = helper::image::readPalettePNG(filenameGt, gt, nullptr);
        if(err != helper::image::PNGError::Okay)
        {
            std::cerr << "Unable to load ground truth \"" << filenameGt << "\". Error Code: " << (int) err << std::endl;
            db->Close();
            return false;
        }
        cv::Mat gt_cv = static_cast<cv::Mat>(gt);
        gt_cv.convertTo(gt_cv, CV_8UC1);

        if(rgb_cv.cols != gt_cv.cols || rgb_cv.rows != gt_cv.rows)
        {
            std::cerr << "RGB patch and GT patch don't match up." << std::endl;
            db->Close();
            return false;
        }

        // Write to database
        caffe::Datum dat;
        dat.set_width(gt_cv.cols);
        dat.set_height(gt_cv.rows);
        dat.set_channels(4); // BGR + Labels
        char* buffer = new char[dat.width() * dat.height() * dat.channels()];
        for (int h = 0; h < dat.height(); ++h)
        {
            for (int w = 0; w < dat.width(); ++w)
            {
                for (int c = 0; c < dat.channels(); ++c)
                {
                    int datum_index = (c * dat.height() + h) * dat.width() + w;
                    if(c < 3)
                        buffer[datum_index] = rgb_cv.at<cv::Vec3b>(h, w)[c];
                    else
                        buffer[datum_index] = gt_cv.at<char>(h, w);
                }
            }
        }
        dat.set_data(buffer, dat.width() * dat.height() * dat.channels());

        std::string out;
        CHECK(dat.SerializeToString(&out));
        transaction->Put(std::to_string(i), out);
        transaction->Commit();
        transaction.reset(db->NewTransaction());

        delete[] buffer;

        std::cout << "OK!" << std::endl;
    }

    transaction->Commit();
    db->Close();

#endif
    return true;
}

bool createFakeMarginals(UtilProperties const& properties)
{
    // Read in file names
    std::vector<std::string> listfile = readLines(properties.job.createFakeMarginals);
    std::cout << listfile.size() << " crops." << std::endl;

    for (std::string const& file : listfile)
    {
        std::string filenameLabeling = file + properties.datasetPx.extension.gt;
        std::string pathLabeling = properties.in + filenameLabeling;
        std::string outPathMarginals = properties.out;

        std::cout << file << " ";

        // Labeling
        LabelImage labeling;
        auto ok = helper::image::readPalettePNG(pathLabeling, labeling, nullptr);
        if (ok != helper::image::PNGError::Okay)
        {
            std::cout << "\tERROR" << std::endl;
            std::cerr << " Couldn't read labeling \"" << pathLabeling << "\". Error Code: " << (int) ok << std::endl;
            return false;
        }

        // Make up crude "marginals"
        FeatureImage marginals(labeling.height(), labeling.width(), properties.datasetPx.constants.numClasses); // All zero

        for (SiteId i = 0; i < marginals.width() * marginals.height(); ++i)
        {
            Label l = labeling.atSite(i);
            if(l < properties.datasetPx.constants.numClasses)
                marginals.atSite(i)[l] = 1;
        }

        if(!marginals.write(outPathMarginals + file + ".mat"))
        {
            std::cout << "\tERROR" << std::endl;
            std::cerr << " Couldn't write marginals to \"" << outPathMarginals << "\"." << std::endl;
            return false;
        }

        std::cout << "OK!" << std::endl;
    }

    return true;
}

bool stitchMarginals(UtilProperties const& properties)
{
    // Read in meta data
    std::vector<std::string> metadata = readLines(properties.job.stitchMarginals);
    std::cout << metadata.size() << " crops." << std::endl;

    std::string const delimiter = ";";
    std::string curImageName;
    int origWidth, origHeight;
    int baseWidth, baseHeight, baseLongSide;
    FeatureImage* pCurStitchedMarginals = nullptr;
    LabelImage curCountImage;

    // Define lambda to finish stitching current marginal map
    auto finishMarginals = [&]()
    {
        if(pCurStitchedMarginals)
        {
            // Average marginals
            for(SiteId i = 0; i < pCurStitchedMarginals->width() * pCurStitchedMarginals->height(); ++i)
            {
                if(curCountImage.atSite(i) > 0)
                {
                    int c = curCountImage.atSite(i);
                    if(c > 0)
                        pCurStitchedMarginals->atSite(i) /= c;
                }
            }

            // Scale to original size
            pCurStitchedMarginals->rescale(origWidth, origHeight, true);
            pCurStitchedMarginals->normalize();

            // Write to hard disk
            std::string outFile = properties.out + curImageName + ".mat";
            if(!pCurStitchedMarginals->write(outFile))
            {
                std::cout << "ERROR" << std::endl;
                std::cerr << "Couldn't write \"" << outFile << "\"." << std::endl;
                delete pCurStitchedMarginals;
            }
            delete pCurStitchedMarginals;
        }
    };

    for(std::string const& meta : metadata)
    {
        // Split info string
        auto start = 0U;
        auto end = meta.find(delimiter);
        std::vector<std::string> tokens;
        while (end != std::string::npos)
        {
            tokens.push_back(meta.substr(start, end - start));
            start = end + delimiter.length();
            end = meta.find(delimiter, start);
        }
        tokens.push_back(meta.substr(start, end));

        if(tokens.size() != 7)
        {
            std::cout << "ERROR" << std::endl;
            std::cerr << "Invalid meta file." << std::endl;
            return false;
        }

        std::string filename = tokens[0];
        int s_x = std::stoi(tokens[1]);
        int s_y = std::stoi(tokens[2]);
        int patchW = std::stoi(tokens[3]);
        int patchH = std::stoi(tokens[4]);
        std::string origFile = tokens[5];
        bool flipped = (tokens[6] == "true");

        std::cout << filename;

        // If this entry is based on a different image, store previous result and create new marginal image
        if(curImageName != origFile)
        {
            // Write out stitched marginals
            finishMarginals();

            // Read in new original image to get dimensions
            RGBImage orig;
            std::string orig_rgb_file = properties.datasetPx.path.rgb_orig + origFile + properties.datasetPx.extension.rgb;
            if(!orig.read(orig_rgb_file))
            {
                std::cout << "ERROR" << std::endl;
                std::cerr << "Couldn't read \"" << orig_rgb_file << "\"." << std::endl;
                return false;
            }

            curImageName = origFile;
            origWidth = orig.width();
            origHeight = orig.height();

            // Compute base size
            int const base_size = properties.prepareDataset.baseSize;
            baseLongSide = base_size + 1;
            baseWidth = baseLongSide;
            baseHeight = baseLongSide;
            if(origHeight > origWidth)
                baseWidth = static_cast<int>(std::round(baseLongSide / (float)origHeight * origWidth));
            else
                baseHeight = static_cast<int>(std::round(baseLongSide / (float)origWidth * origHeight));
            pCurStitchedMarginals = new FeatureImage(baseWidth, baseHeight, properties.datasetPx.constants.numClasses);
            curCountImage = LabelImage(baseWidth, baseHeight);
        }

        // Read in marginals
        FeatureImage marginals;
        if(!marginals.read(properties.in + filename + ".mat"))
        {
            std::cout << "ERROR" << std::endl;
            std::cerr << "Couldn't read \"" << properties.in + filename + ".mat" << "\"." << std::endl;
            return false;
        }

        // Flip back if needed
        if(flipped)
            marginals.flipHorizontally();

        // Scale to original crop size
        marginals.rescale(properties.prepareDataset.cropSize, properties.prepareDataset.cropSize, true);
        marginals.normalize();

        // Copy marginals into destination (this will ignore padded parts)
        pCurStitchedMarginals->addFrom(marginals, s_x, s_y, patchW, patchH);

        // Count pixels that have been touched already
        for(int d_x = s_x; d_x < curCountImage.width() && d_x < s_x + patchW; ++d_x)
        {
            for (int d_y = s_y; d_y < curCountImage.height() && d_y < s_y + patchH; ++d_y)
            {
                curCountImage.at(d_x, d_y)++;
            }
        }

        std::cout << "\tOK!" << std::endl;

    }

    finishMarginals();

    return true;
}

bool createBasicFeatures(UtilProperties const& properties)
{
    // Read in file names
    std::vector<std::string> listfile = readLines(properties.job.createBasicFeatures);
    std::cout << listfile.size() << " crops." << std::endl;

    for (std::string const& file : listfile)
    {
        std::string filenameRgb = file + properties.datasetPx.extension.rgb;
        std::string outPathFeatures = properties.out;

        std::cout << file << " ";

        // Load rgb
        RGBImage rgb;
        if(!rgb.read(properties.in + filenameRgb))
        {
            std::cout << "ERROR" << std::endl;
            std::cerr << "Unable to load rgb image \"" << properties.in + filenameRgb << "\"." << std::endl;
            return false;
        }

        // Convert to CieLAB
        auto cielab = rgb.getCieLabImg();

        // Create feature map
        FeatureImage features(rgb.width(), rgb.height(), 5);
        for(SiteId i = 0; i < rgb.pixels(); ++i)
        {
            auto coords = helper::coord::siteTo2DCoordinate(i, rgb.width());

            // Normalize features into the range [0, 1]
            features.atSite(i)[0] = cielab.atSite(i, 0) / 100.f;
            features.atSite(i)[1] = (cielab.atSite(i, 1) + 127) / (2 * 127 + 1);
            features.atSite(i)[2] = (cielab.atSite(i, 2) + 127) / (2 * 127 + 1);
            features.atSite(i)[3] = coords.x() / static_cast<float>(rgb.width());
            features.atSite(i)[4] = coords.y() / static_cast<float>(rgb.height());
        }

        // Write to disk
        if(!features.write(properties.out + file + properties.datasetPx.extension.img))
        {
            std::cout << "ERROR" << std::endl;
            std::cerr << "Unable to write feature map \"" << properties.out + file + properties.datasetPx.extension.img << "\"." << std::endl;
            return false;
        }

        std::cout << " OK!" << std::endl;
    }
    return true;
}

bool mergeFeatures(UtilProperties const& properties)
{
    // Read in file names
    std::vector<std::string> listfile = readLines(properties.job.mergeFeatures);
    std::cout << listfile.size() << " images." << std::endl;

    for(auto const& filename : listfile)
    {
        std::cout << filename << ": ";

        // Read feature maps
        std::string feat1Filename = properties.mergeFeatures.first + filename + properties.datasetPx.extension.img;
        std::string feat2Filename = properties.mergeFeatures.second + filename + properties.datasetPx.extension.img;
        FeatureImage feat1;
        if (!feat1.read(feat1Filename))
        {
            std::cerr << "Unable to read features from \"" << feat1Filename << "\"" << std::endl;
            return false;
        }

        FeatureImage feat2;
        if (!feat2.read(feat2Filename))
        {
            std::cerr << "Unable to read features from \"" << feat2Filename << "\"" << std::endl;
            return false;
        }

        if(feat1.width() != feat2.width() || feat1.height() != feat2.height())
        {
            std::cerr << "Feature dimensions don't match up!" << std::endl;
            return false;
        }

        FeatureImage res(feat1.width(), feat1.height(), feat1.dim() + feat2.dim());
        for(size_t i = 0; i < res.height() * res.width(); ++i)
        {
            Feature f = Feature::Zero(res.dim());
            f << feat1.atSite(i) , feat2.atSite(i);
            res.atSite(i) = f;
        }

        // Write to disk
        if(!res.write(properties.out + filename + properties.datasetPx.extension.img))
        {
            std::cout << "ERROR" << std::endl;
            std::cerr << "Unable to write feature map \"" << properties.out + filename + properties.datasetPx.extension.img << "\"." << std::endl;
            return false;
        }

        std::cout << " OK!" << std::endl;
    }

    return true;
}

bool testIterationProgress(UtilProperties const& properties)
{
    // Read in file names
    std::vector<std::string> listfile = readLines(properties.job.testIterationProgress);
    std::cout << listfile.size() << " images." << std::endl;

    // Read in weights
    Weights w(properties.datasetPx.constants.numClasses, properties.datasetPx.constants.featDim, properties.datasetCluster.constants.featDim);
    if(!w.read(properties.in))
        std::cout << "Couldn't read in initial weights from \"" << properties.in << "\". Using zero." << std::endl;

    w.printStats(std::cout);
    auto cmap = helper::image::generateColorMapVOC(256);

    // Gather inference data
    std::vector<InferenceResultDetails> results;
    for(auto const& filename : listfile)
    {
        std::cout << filename << ": ";

        // Read images
        std::string imgFilename = properties.datasetPx.path.img + filename + properties.datasetPx.extension.img;
        std::string imgCluFilename = properties.datasetCluster.path.img + filename + properties.datasetCluster.extension.img;
        std::string gtFilename = properties.datasetPx.path.gt + filename + properties.datasetPx.extension.gt;

        FeatureImage featuresPx;
        if(!featuresPx.read(imgFilename))
        {
            std::cerr << "Unable to read features from \"" << imgFilename << "\"" << std::endl;
            return false;
        }

        FeatureImage featuresCluster;
        if(!featuresCluster.read(imgCluFilename))
        {
            std::cerr << "Unable to read features from \"" << imgCluFilename << "\"" << std::endl;
            return false;
        }

        LabelImage gt;
        auto errCode = helper::image::readPalettePNG(gtFilename, gt, nullptr);
        if(errCode != helper::image::PNGError::Okay)
        {
            std::cerr << "Unable to read ground truth from \"" << gtFilename << "\". Error Code: " << (int) errCode << std::endl;
            return false;
        }
        gt.rescale(featuresPx.width(), featuresPx.height(), false);

        // Crop to valid region
        cv::Rect bb = helper::image::computeValidBox(gt, properties.datasetPx.constants.numClasses);
        FeatureImage features_cropped(bb.width, bb.height, featuresPx.dim());
        FeatureImage features_cluster_cropped(bb.width, bb.height, featuresCluster.dim());
        LabelImage gt_cropped(bb.width, bb.height);
        for(Coord x = bb.x; x < bb.width; ++x)
        {
            for (Coord y = bb.y; y < bb.height; ++y)
            {
                gt_cropped.at(x - bb.x, y - bb.y) = gt.at(x, y);
                features_cropped.at(x - bb.x, y - bb.y) = featuresPx.at(x, y);
                features_cluster_cropped.at(x - bb.x, y - bb.y) = featuresCluster.at(x, y);
            }
        }

        gt = gt_cropped;
        featuresPx = features_cropped;
        featuresCluster = features_cluster_cropped;

        if(gt.height() == 0 || gt.width() == 0 || gt.height() != featuresPx.height() || gt.width() != featuresPx.width())
        {
            std::cerr << "Invalid ground truth or features. Dimensions: (" << gt.width() << "x" << gt.height() << ") vs. ("
                      << featuresPx.width() << "x" << featuresPx.height() << ")." << std::endl;
            return false;
        }

        // Predict
        EnergyFunction energy(&w, properties.param.numClusters, properties.param.usePairwise);
        InferenceIterator<EnergyFunction> inference(&energy, &featuresPx, &featuresCluster, properties.param.eps, properties.param.maxIter);
        auto result = inference.runDetailed();

        // Print energies to screen
        for(Cost c : result.energy)
            std::cout << c << ", ";
        std::cout << std::endl;

        // Try to write results to file
        boost::filesystem::path folder = properties.out + filename;

        auto labelingsFolder = folder / "labeling";
        boost::filesystem::create_directories(labelingsFolder);
        for(size_t j = 0; j < result.labelings.size(); ++j)
            helper::image::writePalettePNG(labelingsFolder.string() + "/" + std::to_string(j) + ".png", result.labelings[j], cmap);

        auto clusteringsFolder = folder / "clustering";
        boost::filesystem::create_directories(clusteringsFolder);
        for(size_t j = 0; j < result.clusterings.size(); ++j)
            helper::image::writePalettePNG(clusteringsFolder.string() + "/" + std::to_string(j) + ".png", result.clusterings[j], cmap);

        // Clear memory that is not needed anymore
        result.labelings.clear();
        std::vector<LabelImage>(result.labelings).swap(result.labelings); // Shrink vector to zero
        result.clusterings.clear();
        std::vector<LabelImage>(result.clusterings).swap(result.clusterings);
        result.marginals.clear();
        std::vector<FeatureImage>(result.marginals).swap(result.marginals);
        results.push_back(result);
    }

    // Analyze data
    std::vector<size_t> count;
    std::vector<Cost> meanCostPerIter;
    std::vector<Cost> varCostPerIter;
    // Compute means
    for(auto const& r : results)
    {
        for(size_t i = 0; i < r.energy.size(); ++i)
        {
            if(count.size() <= i)
                count.push_back(1);
            else
                count[i]++;
            if(meanCostPerIter.size() <= i)
                meanCostPerIter.push_back(r.energy[i]);
            else
                meanCostPerIter[i] += r.energy[i];
        }
    }
    for(size_t i = 0; i < count.size(); ++i)
        meanCostPerIter[i] /= count[i];

    // Compute variances
    for(auto const& r : results)
    {
        for(size_t i = 0; i < r.energy.size(); ++i)
        {
            Cost curVal = std::pow(r.energy[i] - meanCostPerIter[i], 2);
            if(varCostPerIter.size() <= i)
                varCostPerIter.push_back(curVal);
            else
                varCostPerIter[i] += curVal;
        }
    }
    for(size_t i = 0; i < count.size(); ++i)
        varCostPerIter[i] /= count[i];

    // Output results
    std::cout << std::setw(4) << "Iter" << "\t;" << std::setw(12) << "Mean" << "\t;" << std::setw(12) << "Variance" << std::endl;
    for(size_t i = 0; i < count.size(); ++i)
    {
        std::cout << std::setw(4) << i << "\t;";
        std::cout << std::setw(12) << meanCostPerIter[i] << "\t;";
        std::cout << std::setw(12) << varCostPerIter[i] << std::endl;
    }

    return true;
}

bool symmetryCheck(UtilProperties const& properties)
{
    Weights w(properties.datasetPx.constants.numClasses, properties.datasetPx.constants.featDim, properties.datasetCluster.constants.featDim);
    if(!w.read(properties.job.symmetryCheck))
    {
        std::cerr << "Couldn't read in weights from \"" << properties.job.symmetryCheck << "\"." << std::endl;
        return false;
    }

    std::cout << "Pairwise: " << w.isPairwiseSymmetric() << std::endl;
    std::cout << "Label Consistency: " << w.isLabelSymmetric() << std::endl;
    std::cout << "Feature: " << w.isFeatureSymmetric() << std::endl;

    return true;
}

bool prepCityscapesGt(UtilProperties const& properties)
{
    // Read in file names
    std::vector<std::string> listfile = readLines(properties.job.prepCityscapesGt);
    std::cout << listfile.size() << " images." << std::endl;

    std::map<Label, Label> labelMap = {
            {0, 29},
            {1, 29},
            {2, 29},
            {3, 29},
            {4, 29},
            {5, 19},
            {6, 20},
            {7, 0},
            {8, 1},
            {9, 21},
            {10, 22},
            {11, 2},
            {12, 3},
            {13, 4},
            {14, 23},
            {15, 24},
            {16, 25},
            {17, 5},
            {18, 29},
            {19, 6},
            {20, 7},
            {21, 8},
            {22, 9},
            {23, 10},
            {24, 11},
            {25, 12},
            {26, 13},
            {27, 14},
            {28, 15},
            {29, 26},
            {30, 27},
            {31, 16},
            {32, 17},
            {33, 18},
            {-1, 28}
    };

    auto cmap = helper::image::generateColorMapCityscapes();

    for(auto const& f : listfile)
    {
        std::string gtFilename = properties.datasetPx.path.gt + f + properties.datasetPx.extension.gt;

        GrayscaleImage gt;
        if(!gt.read(gtFilename))
        {
            std::cerr << "Unable to read \"" << gtFilename << "\"." << std::endl;
            return false;
        }

        LabelImage gt_mapped(gt.width(), gt.height());
        for(SiteId i = 0; i < gt.pixels(); ++i)
            gt_mapped.atSite(i) = labelMap[gt.atSite(i)];

        std::string outFileName = properties.out + f + properties.datasetPx.extension.gt;
        auto errCode = helper::image::writePalettePNG(outFileName, gt_mapped, cmap);
        if(errCode != helper::image::PNGError::Okay)
        {
            std::cerr << "Unable to write to \"" << outFileName << "\". Error Code: " << (int) errCode << std::endl;
            return false;
        }
    }

    return true;
}

bool figureGroundToPascal(UtilProperties const& properties)
{
    // Read in file names
    std::vector<std::string> listfile = readLines(properties.job.figureGroundToPascal);
    std::cout << listfile.size() << " images." << std::endl;

    auto const cmap = helper::image::generateColorMapVOC(256);

    for(auto const& f : listfile)
    {
        std::string fgFilename = properties.in + f + properties.figureGroundToPascal.inExt;
        RGBImage fg_rgb;
        GrayscaleImage fg;
        if(!fg_rgb.read(fgFilename))
        {
            if(!fg.read(fgFilename))
            {
                std::cerr << "Couldn't read \"" << fgFilename << "\"" << std::endl;
                return false;
            }
        }
        else
            fg = fg_rgb.getGrayscaleImg();

        LabelImage gt(fg.width(), fg.height());

        for(SiteId i = 0; i < fg.pixels(); ++i)
        {
            if(fg.atSite(i) < properties.figureGroundToPascal.threshold)
                gt.atSite(i) = properties.figureGroundToPascal.groundLabel;
            else
                gt.atSite(i) = properties.figureGroundToPascal.figureLabel;
        }

        std::string outFileName = properties.out + f + properties.figureGroundToPascal.outExt;
        auto errcode = helper::image::writePalettePNG(outFileName, gt, cmap);
        if(errcode != helper::image::PNGError::Okay)
        {
            std::cerr << "Couldn't write \"" << outFileName << "\". Error Code: " << static_cast<int>(errcode) << std::endl;
            return false;
        }

        std::cout << f << std::endl;
    }
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

    if (!properties.job.maxLoss.empty())
        computeMaxLoss(properties);

    if (!properties.job.outline.empty())
        outline(properties);

    if (!properties.job.rescale.empty())
        rescale(properties);

    if (!properties.job.post_pro.empty())
        post_pro(properties);

    if (!properties.job.matchGt.empty())
        match_gt(properties);

    if(!properties.job.copyFixPNG.empty())
        copyFixPNG(properties);

    if(!properties.job.prepareDataset.empty())
        prepareDataset(properties);

    if(!properties.job.writeLMDB.empty())
        writeLMDB(properties);

    if(!properties.job.createFakeMarginals.empty())
        createFakeMarginals(properties);

    if(!properties.job.stitchMarginals.empty())
        stitchMarginals(properties);

    if(!properties.job.createBasicFeatures.empty())
        createBasicFeatures(properties);

    if(!properties.job.mergeFeatures.empty())
        mergeFeatures(properties);

    if(!properties.job.testIterationProgress.empty())
        testIterationProgress(properties);

    if(!properties.job.symmetryCheck.empty())
        symmetryCheck(properties);

    if(!properties.job.prepCityscapesGt.empty())
        prepCityscapesGt(properties);

    if(!properties.job.figureGroundToPascal.empty())
        figureGroundToPascal(properties);

    return 0;
}