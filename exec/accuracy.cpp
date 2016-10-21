//
// Created by jan on 15.09.16.
//

#include <BaseProperties.h>
#include <helper/image_helper.h>
#include <Accuracy/ConfusionMatrix.h>

PROPERTIES_DEFINE(Accuracy,
                  PROP_DEFINE(std::string, fileList, "")
                  PROP_DEFINE(std::string, predDir, "")
                  PROP_DEFINE(std::string, gtDir, "")
                  PROP_DEFINE(std::string, outDir, "./")
)

std::vector<std::string> readFileNames(std::string const& listFile)
{
    std::vector<std::string> list;
    std::ifstream in(listFile, std::ios::in);
    if (in.is_open())
    {
        std::string line;
        while (std::getline(in, line))
            list.push_back(line);
        in.close();
    }
    return list;
}

enum ErrorCode
{
    ERR_OK = 0,
    ERR_EMPTY_FILE_LIST,
    ERR_IMAGE_LOAD,
    ERR_IMAGE_MISMATCH,
};

int main()
{
    // Read properties
    AccuracyProperties properties;
    properties.read("properties/accuracy.info");
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    auto fileNames = readFileNames(properties.fileList);
    if(fileNames.empty())
    {
        std::cerr << "Empty file list." << std::endl;
        return ERR_EMPTY_FILE_LIST;
    }

    size_t const numClasses = 21ul;
    helper::image::ColorMap const cmap = helper::image::generateColorMapVOC(256ul);
    ConfusionMatrix accuracy(numClasses);
    float loss = 0;

    for(auto const& f : fileNames)
    {
        std::string const& predFilename = properties.predDir + f + ".png";
        std::string const& gtFilename = properties.gtDir + f + ".png";

        // Load images
        RGBImage predRGB;
        predRGB.read(predFilename);
        if (predRGB.pixels() == 0)
        {
            std::cerr << "Couldn't load image " << predFilename << std::endl;
            return ERR_IMAGE_LOAD;
        }
        LabelImage pred = helper::image::decolorize(predRGB, cmap);

        RGBImage gtRGB;
        gtRGB.read(gtFilename);
        if (gtRGB.pixels() == 0)
        {
            std::cerr << "Couldn't load image " << gtFilename << std::endl;
            return ERR_IMAGE_LOAD;
        }
        LabelImage gt = helper::image::decolorize(gtRGB, cmap);

        if(pred.width() != gt.width() || pred.height() != gt.height() || pred.pixels() == 0)
        {
            std::cerr << "Prediction and ground truth don't match." << std::endl;
            return ERR_IMAGE_MISMATCH;
        }

        accuracy.join(pred, gt);

        // Compute loss
        float lossFactor = 0;
        for(size_t i = 0; i < gt.pixels(); ++i)
            if(gt.atSite(i) < numClasses)
                lossFactor++;
        lossFactor = 1e8f / lossFactor;
        for(size_t i = 0; i < gt.pixels(); ++i)
            if (gt.atSite(i) != pred.atSite(i) && gt.atSite(i) < numClasses)
                loss += lossFactor;
    }
    loss /= fileNames.size();

    std::cout << accuracy << std::endl;
    std::cout << "Loss: " << loss << std::endl;
    std::ofstream out(properties.outDir + "accuracy.txt");
    if(out.is_open())
    {
        out << properties << std::endl << std::endl;
        out << accuracy << std::endl;
        out << "Loss: " << loss << std::endl;
        out.close();
    }
    else
        std::cerr << "Couldn't write accuracy to \"" + properties.outDir << "\"" << std::endl;

    cv::Mat confusionMat = static_cast<cv::Mat>(accuracy);
    if(!cv::imwrite(properties.outDir + "confusion.png", confusionMat))
        std::cerr << "Couldn't write confusion matrix to \"" + properties.outDir << "\"" << std::endl;

    return ERR_OK;
}