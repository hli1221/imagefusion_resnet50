%% ResNet50 + ZCA & norm(l1, l2, nuclear)
% load the pre-trained model - ResNet-50
model_path = './models/';
% http://www.vlfeat.org/matconvnet/pretrained/
net_ = load([model_path, 'imagenet-resnet-50-dag.mat']);
net = dagnn.DagNN.loadobj(net_);
net.mode = 'test';
%% remove layers - ResNet50
% Conv5 - res5cx
for i = 173:175
    net.removeLayer(net.layers(173).name);
end
net_res5cx = net;
% Conv4 - res4fx
net = dagnn.DagNN.loadobj(net_);
net.mode = 'test';
for i = 141:175
    net.removeLayer(net.layers(141).name);
end
net_res4fx = net;

%% Start
n = 21; % number of sourc image
time = zeros(n,1);
for i=1:n
    index = i;
    disp(num2str(index));
    
    % infrared and visible images
    path1 = ['./IV_images/IR',num2str(index),'.png'];
    path2 = ['./IV_images/VIS',num2str(index),'.png'];

    % block - 5*5
    % l1 norm
    fuse_path4 = ['./fused_infrared/fused',num2str(index),'_resnet50_l1_zca_4.png'];
    fuse_path5 = ['./fused_infrared/fused',num2str(index),'_resnet50_l1_zca_5.png'];
    
    image1 = imread(path1);
    image2 = imread(path2);
    image1 = im2double(image1);
    image2 = im2double(image2);

    tic;
    %% Extract features, run the net - ResNet50
    disp('ResNet');
    if size(image1, 3)<3
        I1 = make_3c(image1);
    end
    if size(image2, 3)<3
        I2 = make_3c(image2);
    end
    I1 = single(I1) ; % note: 255 range
    I2 = single(I2) ; % note: 255 range

    % I1
    disp('run the ResNet - I1');
    net_res4fx.eval({'data', I1}) ;
    output4_1 = net_res4fx.vars(net_res4fx.getVarIndex('res4fx')).value ;
    net_res5cx.eval({'data', I1}) ;
    output5_1 = net_res5cx.vars(net_res5cx.getVarIndex('res5cx')).value ;
    % I2
    disp('run the ResNet - I2');
    net_res4fx.eval({'data', I2}) ;
    output4_2 = net_res4fx.vars(net_res4fx.getVarIndex('res4fx')).value ;
    net_res5cx.eval({'data', I2}) ;
    output5_2 = net_res5cx.vars(net_res5cx.getVarIndex('res5cx')).value ;

    %% extract features - ZCA & l1-norm operation
    disp('extract features(whitening operation) - I1');
    feature4_1 = whitening_norm(output4_1);
    feature5_1 = whitening_norm(output5_1);
    disp('extract features(whitening operation) - I2');
    feature4_2 = whitening_norm(output4_2);
    feature5_2 = whitening_norm(output5_2);

    %% fusion strategy - resize to original size and soft-max
    disp('fusion strategy(weighting)');
    % output4 - 1024
    [F_relu4, weight4_a, weight4_b] = fusion_strategy(feature4_1, feature4_2, image1, image2);
    % output5 - 2048
    [F_relu5, weight5_a, weight5_b] = fusion_strategy(feature5_1, feature5_2, image1, image2);
    time(i) = toc;

    imwrite(F_relu4,fuse_path4,'png');
    imwrite(F_relu5,fuse_path5,'png');
end


