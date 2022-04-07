%% seven metrics
function metrics = analysis_metrics_7(image_f,image_1,image_2)

[s1,s2] = size(image_1);
imgSeq = zeros(s1, s2, 2);
imgSeq(:, :, 1) = image_1;
imgSeq(:, :, 2) = image_2;

image1 = im2double(image_1);
image2 = im2double(image_2);
image_fused = im2double(image_f);

metrics.EN = entropy(image_fused);
metrics.SD = analysis_sd(image_fused);
metrics.Qabf = Qabf(image1,image2,image_fused);
metrics.VIFF = VIFF_Public(image1,image2,image_fused);
metrics.SSIM = (ssim(image_fused,image1)+ssim(image_fused,image2))/2;


end







