function [ res_arr, err ] = HSI_test( datadir, resdir, fns, methodname )
    num_method = length(methodname);
    num_data = length(fns);
    res_fp = fullfile(resdir, 'res_arr.mat');
    if ~exist(resdir, 'dir')
        mkdir(resdir);
    end
    if ~exist(res_fp, 'file')
        disp('init res_arr ...');
        res_arr = zeros(num_method, num_data, 6); % result hold a table with column names 'psnr', 'ssim', 'fsim', 'ergas' and 'sam'        
    else
        disp('load res_arr ...');
        load(res_fp)
        if size(res_arr, 1) ~= num_method
            res_arr(num_method,1,1) = 0;
        end
        if size(res_arr, 2) ~= num_data
            res_arr(1,num_data,1) = 0;
        end
    end
    
    err = {};
    
    for k = 1:num_data
        for m = [1,2,3,4,6,7,8,9]
%         for m = 1:num_method
%         for m = [2,3,4,5,6,10,13,15]
%         for m = [7,9]
%         for m = [1,2,3,4,5,6,7,9,10,11,12,13,14,15]
            eval_method(m,k);
        end
    end
    
    function [ psnr, ssim, fsim, ergas, sam ] = eval_method( m, k )
        fn = fns{k};
        method = methodname{m};
        if abs(res_arr(m,k,1)) > 1e-5  % result reuse
            disp(['reuse precomputed result in pos (' num2str([m k]) ')']);
            fprintf('psnr: %.3f\n', res_arr(m,k,1));
            fprintf('ssim: %.3f\n', res_arr(m,k,2));
        else
            disp(['perform ' method ' in pos (' num2str([m k]) ')' ]);
            filepath = fullfile(datadir, fn);
            [~, imgname] = fileparts(fn);

            mat = load(filepath); % contain (input, gt, sigma)        
            hsi = mat.gt;
            noisy_hsi = mat.input;

            if ~isfield(mat, 'sigma')
                sigma_ratio = NoiseLevel(noisy_hsi);  
            else
                sigma_ratio = mat.sigma / 255;
            end
            
            imgdir = fullfile(resdir, imgname);
            if ~exist(imgdir, 'dir')
                mkdir(imgdir);
            end
            
            try
                [R_hsi, time] = demo_fun(noisy_hsi, sigma_ratio, method);                                                
                save(fullfile(imgdir, method), 'R_hsi');
                [psnr, ssim, fsim, ergas, sam] = MSIQA(hsi*255, R_hsi*255);
                fprintf('psnr: %.3f\n', psnr);
                fprintf('ssim: %.3f\n', ssim);
                res_arr(m, k, :) = [psnr, ssim, fsim, ergas, sam, time];
                save(res_fp, 'res_arr');
            catch Error
                disp(['error occured in ' [m k]]);
                disp(Error);
                err{end+1} = [m k];
            end
        end                        
    end
end

