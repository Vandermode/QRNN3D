function [ ] = generate_dataset_blind( datadir, fns, newdir, gt_key, preprocess )
%GENERATE_DATASET Summary of this function goes here
    
    if ~exist(newdir, 'dir')
        mkdir(newdir)
    end
        
    for k = 1:length(fns)
        fn = fns{k};
        fprintf('generate data(%d/%d)\n', k, length(fns));
        filepath = fullfile(datadir, fn);
        mat = load(filepath); % contain gt_key
        gt = getfield(mat, gt_key);
        
        if exist('preprocess', 'var')
            gt = preprocess(gt);
        end
        
        gt = normalized(gt);
        sigma = rand(1,1) * 60 + 10;
%         sigma = rand(1,1) * 85 + 15;
        fprintf('sigma: %.2f\n', sigma);
        s = reshape(sigma, 1, 1, length(sigma));
        input = gt + s/255 .* randn(size(gt));
        save(fullfile(newdir, fn), 'gt', 'input', 'sigma');
    end
end
