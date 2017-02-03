//
// Created by jan on 03.02.17.
//

#include <BaseProperties.h>
#include <caffe/caffe.hpp>
#include <opencv2/core/types.hpp>
#include <Image/Image.h>
#include <helper/image_helper.h>
#include <helper/coordinate_helper.h>

PROPERTIES_DEFINE(TrainFeat,
)

cv::Mat forward(caffe::Net<float>& net, cv::Mat patch)
{
    caffe::Blob<float>* input_layer = net.input_blobs()[0];
    caffe::Blob<float>* output_layer = net.output_blobs()[0];

    CHECK(patch.cols == input_layer->width() && patch.rows == input_layer->height() && patch.channels() == input_layer->channels())
    << "Patch doesn't have the right dimensions.";

    input_layer->Reshape(1, 3, patch.rows, patch.cols);
    net.Reshape();
    std::vector<cv::Mat> input_channels;
    float* input_data = input_layer->mutable_cpu_data();
    for (int i = 0; i < input_layer->channels(); ++i)
    {
        cv::Mat channel(input_layer->height(), input_layer->width(), CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += input_layer->width() * input_layer->height();
    }
    cv::split(patch, input_channels);

    CHECK(reinterpret_cast<float*>(input_channels.at(0).data) == net.input_blobs()[0]->cpu_data())
    << "Input channels are not wrapping the input layer of the network.";

    net.ForwardPrefilled();

    // Copy results back
    const float* begin = output_layer->cpu_data();
    cv::Mat scores(output_layer->height(), output_layer->width(), CV_32FC(output_layer->channels()));
    for(int y = 0; y < scores.rows; ++y)
    {
        for(int x = 0; x < scores.cols; ++x)
        {
            for(int c = 0; c < scores.channels(); ++c)
                scores.ptr<float>(y)[scores.channels()*x+c] = *(begin + (x + y * output_layer->width() + c * output_layer->width() * output_layer->height()));
        }
    }

    return scores;
}

int main(int argc, char** argv)
{
    // Read properties
    TrainFeatProperties properties;
    properties.read("properties/hseg_train_feat.info");
    properties.fromCmd(argc, argv);
    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Used properties: " << std::endl;
    std::cout << properties << std::endl;
    std::cout << "----------------------------------------------------------------" << std::endl;

    // Setup protobuf logging
    ::google::InitGoogleLogging(argv[0]);

    // Init network
    caffe::Net<float> net("/home/jan/Dokumente/Git/hseg/data/net/prototxt/pspnet101_VOC2012_473.prototxt.bak", caffe::Phase::TEST);
    net.CopyTrainedLayersFrom("/home/jan/Dokumente/Git/hseg/data/net/model/pspnet101_VOC2012.caffemodel");

    std::cout << "#in: " << net.num_inputs() << std::endl;
    std::cout << "#out: " << net.num_outputs() << std::endl;

    caffe::Blob<float>* input_layer = net.input_blobs()[0];
    std::cout << "in_channels: " << input_layer->channels() << std::endl;
    std::cout << "in_width: " << input_layer->width() << std::endl;
    std::cout << "in_height: " << input_layer->height() << std::endl;

    caffe::Blob<float>* output_layer = net.output_blobs()[0];
    std::cout << "out_channels: " << output_layer->channels() << std::endl;
    std::cout << "out_width: " << output_layer->width() << std::endl;
    std::cout << "out_height: " << output_layer->height() << std::endl;

    // Load an image
    RGBImage rgb;
    rgb.read("/home/jan/Downloads/Pascal VOC/data/VOC2012/JPEGImages/2007_000032.jpg");
    cv::Mat rgb_cv = static_cast<cv::Mat>(rgb);
//    cv::cvtColor(rgb_cv, rgb_cv, CV_BGR2RGB);
    rgb_cv.convertTo(rgb_cv, CV_32FC3);

    std::cout << "im_channels: " << rgb_cv.channels() << std::endl;
    std::cout << "im_width: " << rgb_cv.cols << std::endl;
    std::cout << "im_height: " << rgb_cv.rows << std::endl;

    // Crop out a part that has the right dimensions
    unsigned const int x = 0;
    unsigned const int y = 0;
    unsigned int patch_w = input_layer->width();
    unsigned int patch_h =  input_layer->height();
    if(static_cast<int>(x + patch_w) > rgb_cv.cols)
        patch_w = rgb_cv.cols - x;
    if(static_cast<int>(y + patch_h) > rgb_cv.rows)
        patch_h = rgb_cv.rows - y;
    cv::Mat patch = rgb_cv(cv::Rect(x, y, patch_w, patch_h));

    // Subtract mean
    float const mean_r = 123.680f;
    float const mean_g = 116.779f;
    float const mean_b = 103.939f;

    cv::Mat channels[3];
    cv::split(patch, channels);
    channels[2] -= mean_r;
    channels[1] -= mean_g;
    channels[0] -= mean_b;
    cv::Mat normalized_img(patch.rows, patch.cols, CV_32FC3);
    cv::merge(channels, 3, normalized_img);

    // Pad with zeros
    unsigned int const pad_w = input_layer->width() - patch_w;
    unsigned int const pad_h = input_layer->height() - patch_h;
    cv::Mat padded_img;
    cv::copyMakeBorder(normalized_img, padded_img, 0, pad_h, 0, pad_w, cv::BORDER_CONSTANT, cv::Scalar(0));

    // Run it through the network
    auto scores = forward(net, padded_img);
//    cv::flip(padded_img, padded_img, 0);
//    auto scores_flip = forward(net, padded_img);
//    cv::flip(padded_img, padded_img, 0);


    // Show labeling
    LabelImage labeling(output_layer->width(), output_layer->height());
    for(int y = 0; y < scores.rows; ++y)
    {
        for(int x = 0; x < scores.cols; ++x)
        {
            float maxCost = scores.ptr<float>(y)[scores.channels() * x + 0];
            Label maxLabel = 0;
            for (Label l = 1; l < 21; ++l)
            {
                float cost = scores.ptr<float>(y)[scores.channels() * x + l];
                if (cost > maxCost)
                {
                    maxCost = cost;
                    maxLabel = l;
                }
            }
            labeling.at(x, y) = maxLabel;
        }
    }

    auto cmap = helper::image::generateColorMapVOC(256);

    cv::imshow("patch", patch / 255);
    double min, max;
    cv::minMaxLoc(padded_img.reshape(1, 1), &min, &max);
    cv::imshow("patch padded", (padded_img - min) / (max - min));
    cv::imshow("labeling", (cv::Mat)helper::image::colorize(labeling, cmap));
    cv::waitKey();


    return 0;
}