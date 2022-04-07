% analysis the fused image
clear all;
clc;
addpath('./vif');
disp('----------analysis_main_infrared_average------------');

% types = {'_TA_cGAN', '_PA_PCNN', '_CNNs', '_NDM', '_u2fuse', '_CTD_SR', '_my_test', '_LLF_IOI'};
% % please replace the image paths
% for type = 1:length(types)
%     values = [0, 0, 0, 0, 0];
%     for i = 1:10
%         fileName_source1 = strcat('C:/Users/25718/Desktop/MR_FDG/MR_FDG_png/norm_brain/MR_T1/', num2str(i), '_MR_T10.png');
%         fileName_source2 = strcat('C:/Users/25718/Desktop/MR_FDG/MR_FDG_png/norm_brain/PET/', num2str(i), '_FDG0.png');
%         fileName_fused = strcat('./Fused_images/MR_T1_PET/', num2str(i), types(type), '.png');
%         
%         % 融合结果
%         fusedImage = imread(string(fileName_fused(1)));
%         % 输入图
%         sourceImage1 = imread(fileName_source1);
%         sourceImage2 = imread(fileName_source2);
%     
%         fusedImage = rgb2gray(fusedImage);
% %     sourceImage1 = rgb2gray(sourceImage1);
%     
%     % tic;
%         metrics = analysis_metrics_7(fusedImage,sourceImage1,sourceImage2);
%     % toc;
%     % EN, SD, Qabf, SSIM, VIFF
%         values = values + [metrics.EN, metrics.SD, metrics.Qabf, metrics.VIFF, metrics.SSIM];
% 
%         disp(['EN:', num2str(metrics.EN), ', SD:', num2str(metrics.SD),  ...
%              ', Qabf:', num2str(metrics.Qabf), ', VIFF:', num2str(metrics.VIFF), ', SSIM:', num2str(metrics.SSIM)]);
%     end
%     disp(['processing...', string(types(type))]);
%     disp(values ./ 10);
% end

values = [0, 0, 0, 0, 0];
    for i = 1:10
        fileName_source1 = strcat('C:/Users/25718/Desktop/MR_FDG/MR_FDG_png/norm_brain/MR_T1/', num2str(i), '_MR_T10.png');
        fileName_source2 = strcat('C:/Users/25718/Desktop/MR_FDG/MR_FDG_png/norm_brain/PET/', num2str(i), '_FDG0.png');
        fileName_fused = strcat('C:/Users/25718/Desktop/MR_FDG/metrics/Fused_images/ablation/', num2str(i), 'no',  '.png');
        
        % 融合结果
        fusedImage = imread(string(fileName_fused));
        % 输入图
        sourceImage1 = imread(fileName_source1);
        sourceImage2 = imread(fileName_source2);
    
        fusedImage = rgb2gray(fusedImage);
%     sourceImage1 = rgb2gray(sourceImage1);
    
    % tic;
        metrics = analysis_metrics_7(fusedImage,sourceImage1,sourceImage2);
    % toc;
    % EN, SD, Qabf, SSIM, VIFF
        values = values + [metrics.EN, metrics.SD, metrics.Qabf, metrics.VIFF, metrics.SSIM];

        disp(['EN:', num2str(metrics.EN), ', SD:', num2str(metrics.SD),  ...
             ', Qabf:', num2str(metrics.Qabf), ', VIFF:', num2str(metrics.VIFF), ', SSIM:', num2str(metrics.SSIM)]);
    end
%     disp(['processing...', string(types(type))]);
    disp(values ./ 10);

%  values = [0, 0, 0, 0, 0];
%  fileName_source1 = 'C:/Users/25718/Desktop/MR_FDG/MR_FDG_png/AIDS/MR_PD/7_MR_PD0.png';
%  fileName_source2 = 'C:/Users/25718/Desktop/MR_FDG/MR_FDG_png/AIDS/SPECT/7_SPECT_TC0.png';
%  fileName_fused = 'G:/Stone/imagefusion-nestfuse-master_gai/imagefusion-nestfuse-master/addition/7_TA1-1000.png.png';
%         
%         % 融合结果
%  fusedImage = imread(fileName_fused);
%         % 输入图
%  sourceImage1 = imread(fileName_source1);
%  sourceImage2 = imread(fileName_source2);
%     
%  fusedImage = rgb2gray(fusedImage);
% %     sourceImage1 = rgb2gray(sourceImage1);
%     
%     % tic;
%  metrics = analysis_metrics_7(fusedImage,sourceImage1,sourceImage2);
%     % toc;
%     % EN, SD, Qabf, SSIM, VIFF
%  values = values + [metrics.EN, metrics.SD, metrics.Qabf, metrics.VIFF, metrics.SSIM];
% 
%  disp(['EN:', num2str(metrics.EN), ', SD:', num2str(metrics.SD),  ...
%              ', Qabf:', num2str(metrics.Qabf), ', VIFF:', num2str(metrics.VIFF), ', SSIM:', num2str(metrics.SSIM)]);
% 
