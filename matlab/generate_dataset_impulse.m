function [ ] = generate_dataset_impulse( datadir, fns, newdir, sigmas, gt_key, preprocess )
%GENERATE_DATASET Summary of this function goes here    
    ratios = [0.1, 0.3, 0.5, 0.7];
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
        % sample sigma uniformly from sigmas
        idx = randi(length(sigmas), size(gt,3), 1);
        sigma = sigmas(idx);
        disp(sigma)
        
        s = reshape(sigma, 1, 1, length(sigma));
        input = gt + s/255 .* randn(size(gt));
                
        [~, N, B] = size(gt);
        
        band = randperm(B);
        band = band(1:10);
        idx = randi(length(ratios), length(band), 1);
        ratio = ratios(idx);
        disp(ratio);

        for i=1:length(band)
            input(:,:,band(i)) = imnoise(input(:,:,band(i)),'salt & pepper',ratio(i));
        end
                
        save(fullfile(newdir, fn), 'gt', 'input', 'sigma');
    end
end
