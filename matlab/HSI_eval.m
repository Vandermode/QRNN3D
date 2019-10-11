function [ res_arr ] = HSI_eval( datadir, resdir, fns, method )
    num_data = length(fns);
    res_arr = zeros(num_data, 5);
    for k = 1:num_data
        fn = fns{k};
        disp(['evaluate ' method ' in pos (' num2str(k) ')' ]);        
        [~, imgname] = fileparts(fn);        
        filepath = fullfile(datadir, fn);
        mat = load(filepath); % contain (input, gt, sigma)        
        hsi = mat.gt;
        imgdir = fullfile(resdir, imgname);                
        load(fullfile(imgdir, method)); % load R_hsi
        [psnr, ssim, fsim, ergas, sam] = MSIQA(hsi*255, R_hsi*255);
        fprintf('psnr: %.3f\n', psnr);
        fprintf('ssim: %.3f\n', ssim);
        res_arr(k, :) = [psnr, ssim, fsim, ergas, sam];         
    end  

end

