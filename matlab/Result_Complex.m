%% Result Analysis
addpath(genpath('lib'));

basedir = 'Data';
methodname = {'None','BWBM3D','BM4D','LRMR','LRTV','NMoG','LRTDTV','HSID-CNN', 'MemNet','QRNN3D'};
num_method = length(methodname);
ds_names = {'icvl_512_noniid', 'icvl_512_stripe', 'icvl_512_deadline', 'icvl_512_impulse', 'icvl_512_mixture'};
titles = {'non-i.i.d.', 'stripe', 'deadline', 'impulse', 'mixture'};
ms = 1:num_method; % displayed method

% columnLabels= methodname(ms);
rowLabels = {'PSNR', 'SSIM', 'SAM'};

g1 = load(fullfile(basedir, '_meta_complex.mat')); % load fns
g2 = load(fullfile(basedir, '_meta_complex_2.mat'));
fns = [g1.fns;g2.fns];

%%

for d = 1%[1,2,3,4,5]
    dataset_name = ds_names{d};    
    resdir = fullfile('Result', dataset_name);
    load(fullfile(resdir, 'res_arr_new'));  % load res_arr        
    res_table = zeros(length(ms), 5, 2);
    
    disp(['============= ' dataset_name ' ============='])
    for i = 1:length(ms)
        m = ms(i);
        disp(['============= ' methodname{m} ' ============='])
        psnr = nonzeros(res_arr(m,:,1)); 
        ssim = nonzeros(res_arr(m,:,2)); 
%         fsim = nonzeros(res_arr(m,:,3)); 
%         ergas = nonzeros(res_arr(m,:,4)); 
        sam = nonzeros(res_arr(m,:,5)); 

        res_table(i, 1, 1) = mean(psnr);
        res_table(i, 2, 1) = mean(ssim);
%         res_table(i, 3, 1) = mean(fsim);
%         res_table(i, 4, 1) = mean(ergas);
        res_table(i, 5, 1) = mean(sam);
        res_table(i, 1, 2) = std(psnr);
        res_table(i, 2, 2) = std(ssim);
%         res_table(i, 3, 2) = std(fsim);
%         res_table(i, 4, 2) = std(ergas);
        res_table(i, 5, 2) = std(sam);

        disp(['PSNR:  ' num2str(mean(psnr)) '( ' num2str(std(psnr)) ' )']);
        disp(['SSIM:  ' num2str(mean(ssim)) '( ' num2str(std(ssim)) ' )']);
%         disp(['FSIM:  ' num2str(mean(fsim)) '( ' num2str(std(fsim)) ' )']);
%         disp(['ERGAS: ' num2str(mean(ergas)) '( ' num2str(std(ergas)) ' )']);
        disp(['SAM: ' num2str(mean(sam)) '( ' num2str(std(sam)) ' )']);
    end
end
