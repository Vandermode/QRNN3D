function [ ] = generate_dataset_complex( noise_type, datadir, fns, newdir, sigmas, gt_key, preprocess )
%GENERATE_DATASET Summary of this function goes here    
    if strcmp(noise_type, 'noniid')
        generate_dataset_noniid( datadir, fns, newdir, sigmas, gt_key, preprocess )
    elseif strcmp(noise_type, 'stripe')
        generate_dataset_stripe( datadir, fns, newdir, sigmas, gt_key, preprocess )
    elseif strcmp(noise_type, 'deadline')
        generate_dataset_deadline( datadir, fns, newdir, sigmas, gt_key, preprocess )
    elseif strcmp(noise_type, 'impulse')
        generate_dataset_impulse( datadir, fns, newdir, sigmas, gt_key, preprocess )
    elseif strcmp(noise_type, 'mixture')
        generate_dataset_mixture( datadir, fns, newdir, sigmas, gt_key, preprocess )
    end
end
