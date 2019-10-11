%==========================================================================
% clc;
clear;
close all;
addpath(genpath('lib'));

%% Data init
basedir = 'Data';
g1 = load(fullfile(basedir, '_meta_complex.mat')); % load fns
g2 = load(fullfile(basedir, '_meta_complex_2.mat')); % load fns
fns = [g1.fns;g2.fns];
methodname = {'None','BWBM3D','BM4D','LRMR','LRTV','NMoG','LRTDTV'};


%%

ds_names = {'icvl_512_noniid', 'icvl_512_stripe', 'icvl_512_deadline', 'icvl_512_impulse', 'icvl_512_mixture'};

for i = 1:5
    dataset_name = ds_names{i};
    datadir = fullfile(basedir, dataset_name);
    resdir = fullfile('Result', dataset_name);
    HSI_test(datadir, resdir, fns, methodname);
end

