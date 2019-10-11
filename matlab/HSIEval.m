%% Evaluate the result based on Result w.r.t. 5 PQIs 
basedir = 'Data';
method = 'QRNN3D';
% ds_names = {'icvl_512_10', 'icvl_512_30', 'icvl_512_50', 'icvl_512_70', 'icvl_512_blind'};
ds_names = {'icvl_512_noniid', 'icvl_512_stripe', 'icvl_512_deadline', 'icvl_512_impulse', 'icvl_512_mixture'};

% g1 = load(fullfile(basedir, '_meta_gauss.mat')); % load fns
% g2 = load(fullfile(basedir, '_meta_gauss_2.mat'));
g1 = load(fullfile(basedir, '_meta_complex.mat')); % load fns
g2 = load(fullfile(basedir, '_meta_complex_2.mat')); % load fns
fns = [g1.fns;g2.fns];

% methodname = {'None','LRMR','LRTV','NMoG','LRTDTV', 'HSID-CNN', 'MemNet', 'QRNN3D'};
extra_methods = {'HSID-CNN', 'MemNet', 'QRNN3D'};

%%
% fns = {'PaviaU.mat'};
% dataset_name = 'Pavia_mixture_full';
% datadir = fullfile(basedir, dataset_name);
% resdir = fullfile('Result', dataset_name);
% extra_res = ECCV_eval(datadir, resdir, fns, method);        
% load(fullfile(resdir, 'res_arr_final')); % load res_arr
% res_arr(end+1, :,1:5) = extra_res;
% save(fullfile(resdir, 'res_arr_final2'), 'res_arr');
% clear res_arr

% methodname = {'None','LRMR','LRTV','NMoG','LRTDTV', 'HSID-CNN', 'QRNN3D-P', 'QRNN3D-F'};
%%

% res_arr = zeros(length(methodname), 1, 5);
% for i = 1:length(methodname)
%     method = methodname{i};
%     res = ECCV_eval(datadir, resdir, fns, method);    
%     res_arr(i,:,:) = res;
% end
% 
% save(fullfile(resdir, 'res_arr'),'res_arr');

for i = 1:5
    dataset_name = ds_names{i};
    datadir = fullfile(basedir, dataset_name);
    resdir = fullfile('Result', dataset_name);
    load(fullfile(resdir, 'res_arr')); % load res_arr
    for k=1:length(extra_methods)
        method = extra_methods{k};
        extra_res = HSI_eval(datadir, resdir, fns, method);        
        
        res_arr(end+1, :,:) = extra_res;
        save(fullfile(resdir, 'res_arr_new'), 'res_arr');
    end
end