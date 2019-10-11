%==========================================================================
% clc;
clear;
close all;
addpath(genpath('lib'));

%% Data init
basedir = 'Data';
g1 = load(fullfile(basedir, '_meta_gauss.mat')); % load fns
g2 = load(fullfile(basedir, '_meta_gauss_2.mat'));
fns = [g1.fns;g2.fns];

methodname = {'None','BM4D','TDL', 'ITSReg','LLRT'};

if isempty(gcp)
    parpool(4,'IdleTimeout', inf); % If your computer's memory is less than 8G, do not use more than 4 workers.
end

ds_names = {'icvl_512_30', 'icvl_512_50', 'icvl_512_70', 'icvl_512_blind'};

for i = 4
    dataset_name = ds_names{i};
    datadir = fullfile(basedir, dataset_name);
    resdir = fullfile('Result', dataset_name);
    HSI_test(datadir, resdir, fns, methodname);
end
