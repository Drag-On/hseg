close all; clc; clear;
addpath('../../PSPNet/matlab'); %add matcaffe path
%addpath('../visualizationCode');

data_root = '../data/PascalVOC2012/train/rgb';
eval_list = '../data/PascalVOC2012/train/metadata.txt';
save_root = '../data/PascalVOC2012/train/feat';
model_weights = 'model/pspnet101_VOC2012.caffemodel';
model_deploy = 'prototxt/deploy.prototxt';
crop_size = 473;
use_gpu = false;
gpu_id = 0;
img_ext = '.jpg';

mean_r = 123.68; %means to be subtracted and the given values are used in our training stage
mean_g = 116.779;
mean_b = 103.939;

list = importdata(eval_list);

if(~isdir(save_root))
    mkdir(save_root);
end

phase = 'test'; %run with phase test (so that dropout isn't applied)
if ~exist(model_weights, 'file')
  error('Model missing!');
end
if ~exist(model_deploy, 'file')
  error('Prototxt missing!');
end
caffe.reset_all();
if use_gpu
	caffe.set_mode_gpu();
	caffe.set_device(gpu_id);
end
net = caffe.Net(model_deploy, model_weights, phase);

for i=1:length(list)
	elem = list{i};
	str = strsplit(elem,';');
	str = str{1};

	fprintf('%s',str);

    if exist([save_root '/' str '.mat'], 'file')
        fprintf(' exists\n');
        continue
    end

    img = imread(fullfile(data_root,strcat(str,img_ext)));

    [w, h, c] = size(img);
    if w ~= crop_size or h ~= crop_size or c ~= 3
    	error('Image dimensions don''t fit!');
    end

    im_mean = zeros(crop_size,crop_size,3,'single');
    im_mean(:,:,1) = mean_r;
    im_mean(:,:,2) = mean_g;
    im_mean(:,:,3) = mean_b;
    img = single(img) - im_mean;

    img = permute(img,[2 1 3]);

    features = net.forward({img});
    features = features{1};

    features = permute(features, [2 1 3]);

    save([save_root '/' str],'features');

    fprintf(' OK\n');
end