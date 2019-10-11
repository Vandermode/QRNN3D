addpath(genpath('lib'));
% datadir = fullfile('Data','Indian');
datadir = fullfile('Data','Urban');
% datadir = fullfile('Data','Harvard');

% resdir = fullfile('Result', 'Indian');
resdir = fullfile('Result', 'Urban');
% resdir = fullfile('Result', 'Harvard');

% fns = {'Indian_pines.mat'};
fns = {'Urban183.mat'};
% fns = {'img1.mat'};
methodname = {'None', 'BM4D','TDL', 'ITSReg', 'LLRT','LRMR','LRTV','NMoG','LRTDTV'};

for m = 1:length(methodname)
    method = methodname{m};
    for k = 1:length(fns)
        fn = fns{k};
        [~, imgname] = fileparts(fn);
        imgdir = fullfile(resdir, imgname);
        savepath = fullfile(imgdir, [methodname{m}, '.mat']);
        if ~exist(imgdir, 'dir')
            mkdir(imgdir);
        end
        if exist(savepath, 'file')
            disp(['reuse precomputed result in pos (' num2str([m k]) ')']);
            break
        end
        load(fullfile(datadir, fn))
        sigma_ratio = double(real(NoiseLevel(hsi))); 
        disp(['perform ' method ' in pos (' num2str([m k]) ')' ]);        
        R_hsi = demo_fun(hsi, sigma_ratio, methodname{m});
        save(savepath, 'R_hsi');
    end
end



