function [ PSNR, SSIM, FSIM, ERGAS ] = eval_dataset( datadir, fns, method, preprocess )
%EVAL_DATASET Summary of this function goes here    
    
    for k = 1:length(fns)
        fn = fns{k};
        disp(['eval ' fn]);
        filepath = fullfile(datadir, fn);
        mat = load(filepath); % contain (input, gt, sigma)        
        hsi = mat.gt;
        noisy_hsi = mat.input;
        if exist('preprocess', 'var')
            hsi = preprocess(hsi);
            noisy_hsi = preprocess(noisy_hsi);
        end
        if ~isfield(mat, 'sigma')
            sigma_ratio = NoiseLevel(noisy_hsi);  
        else
            sigma_ratio = mat.sigma / 255;
        end
        
        R_hsi = demo_fun(noisy_hsi, sigma_ratio, method);
        [psnr, ssim, fsim, ergas] = MSIQA(hsi*255, R_hsi*255);
        fprintf('psnr: %.3f\n', psnr);
        fprintf('ssim: %.3f\n', ssim);
        PSNR(k) = psnr;
        SSIM(k) = ssim;
        FSIM(k) = fsim;
        ERGAS(k) = ergas;
    end
    disp(mean(PSNR));
end

