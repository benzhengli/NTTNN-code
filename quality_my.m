function [MPSNR, MSSIM] = quality_my(imagery1, imagery2)
Nway = size(imagery1);
PSNR = 0;
SSIM = PSNR;
for i = 1:Nway(3)
    PSNR = psnr(imagery2(:, :, i), imagery1(:, :, i),255) + PSNR;
    SSIM = ssim(imagery2(:, :, i)./255, imagery1(:, :, i)./255) + SSIM;
end
MPSNR = PSNR / Nway(3);
MSSIM = SSIM / Nway(3);


