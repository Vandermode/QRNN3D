function [ Re_hsi, time ] = demo_fun( noisy_hsi, sigma_ratio, methodname )
%DEMO_FUN
hsi_sz  =  size(noisy_hsi);
Re_hsi  =  noisy_hsi;
tic;
if strcmp(methodname, 'None')
%     Re_hsi(Re_hsi>1) = 1;
%     Re_hsi(Re_hsi<0) = 0;
    time = toc;
    return

elseif strcmp(methodname, 'BWBM3D')
    if length(sigma_ratio) == 1
        sigma_ratio = repmat(sigma_ratio, hsi_sz(3), 1);
    end
    for ch = 1:hsi_sz(3)
        [~, Re_hsi(:, :, ch)] = BM3D(1, noisy_hsi(:, :, ch), sigma_ratio(ch)*255);
    end
    
elseif strcmp(methodname, 'TDL')
    vstbmtf_params.peak_value = 1;
    vstbmtf_params.nsigma = mean(sigma_ratio);
    Re_hsi = TensorDL(noisy_hsi, vstbmtf_params);

elseif strcmp(methodname, 'BM4D')
    if length(sigma_ratio) == 1
        [~, Re_hsi] = bm4d(1, noisy_hsi, sigma_ratio);
    else
%         [~, Re_hsi] = bm4d(1, noisy_hsi, 0);  % enable automatical sigma estimation
        [~, Re_hsi] = bm4d(1, noisy_hsi, mean(sigma_ratio));
    end

elseif strcmp(methodname, 'ITSReg')
    Re_hsi = ITS_DeNoising(noisy_hsi,mean(sigma_ratio), 1);

elseif strcmp(methodname, 'BCTF-HSI')
    Re_hsi = BCTF_DeNoising(noisy_hsi,mean(sigma_ratio)*255);
    
elseif strcmp(methodname, 'BCTF')
    model = BCPF(noisy_hsi, 'init', 'rand', 'maxRank', 30, 'dimRed', 1, 'tol', 1e-3, 'maxiters', 25, 'verbose', 0);
    Re_hsi = double(model.X);
    
elseif strcmp(methodname, 'LLRT')
    Par = LLRT_ParSet(mean(sigma_ratio)*255);
    Re_hsi = LLRT_DeNoising(noisy_hsi*255, Par)/255 ;

elseif strcmp(methodname, 'NMoG')
    if size(noisy_hsi, 3) > 100
        r = 5; % objective rank of low rank component
        param.initial_rank = 30;      
        param.rankDeRate = 7;        
        param.mog_k = 5;             
        param.lr_init = 'SVD';
        param.maxiter = 30;
        param.tol = 1e-4;
        param.display = 0;
    else
        r = 3;
        param.initial_rank = 30;      % initial rank of low rank component
        param.rankDeRate = 7;         % the number of rank reduced in each iteration
        param.mog_k = 3;              % the number of component reduced in each band
        param.lr_init = 'SVD';
        param.maxiter = 30;
        param.tol = 1e-3;
        param.display = 0;
    end
    Re_hsi = NMoG(noisy_hsi, r, param);

elseif strcmp(methodname, 'LRTV')    
    if size(noisy_hsi,3) > 100
        tau = 0.015;
        lambda = 20/sqrt(hsi_sz(1)*hsi_sz(2));
        rank = 10;
    else
%         ICVL
        tau = 0.01;
        lambda = 10/sqrt(hsi_sz(1)*hsi_sz(2));
        rank = 5;
    end
    Re_hsi = LRTV(noisy_hsi, tau, lambda, rank);

elseif strcmp(methodname, 'LRMR')
    if size(noisy_hsi, 3) > 100
        r = 7;
        slide =20;
        s = 0.1;
        stepsize = 8;
    else
        % ICVL
%         r = 2;
%         slide = 30;
%         s = 0.05;
%         stepsize = 4;
        % CAVE
        r = 3;
        slide = 20;
        s = 0.00;
        stepsize = 4;
    end
    Re_hsi = LRMR_HSI_denoise( noisy_hsi,r,slide,s,stepsize );
    
elseif strcmp(methodname, 'LRTDTV')
    if size(noisy_hsi,3) > 100
        Re_hsi = LRTDTV(noisy_hsi, 1, 10, [ceil(0.8*hsi_sz(1)), ceil(0.8*hsi_sz(2)), 10]);
    else
        Re_hsi = LRTDTV(noisy_hsi, 1, 10, [ceil(0.1*hsi_sz(1)), ceil(0.1*hsi_sz(2)), 3]);
    end
    
elseif strcmp(methodname, 'LRTA')
    Re_hsi = double(LRTA(tensor(noisy_hsi)));   
elseif strcmp(methodname, 'PARAFAC')    
    if size(noisy_hsi,3) > 100
        Re_hsi = PARAFAC(tensor(double(noisy_hsi)), 2e-6, 2e-5);
    else
        Re_hsi = PARAFAC(tensor(double(noisy_hsi)));    
    end
elseif strcmp(methodname, 'tSVD')
    Re_hsi = tSVD_DeNoising(noisy_hsi,mean(sigma_ratio), 1);
elseif strcmp(methodname, 'LLRGTV')
    if size(noisy_hsi,3) > 100
        par.lambda = 0.20;
        par.tau  = 0.005;
        par.r = 2;
        par.blocksize = 20;
        par.stepsize  = 10;
        par.maxIter = 50;
        par.tol = 1e-6;
    else        
        par.lambda = 0.13;
        par.tau  = 0.013;
        par.r = 2;
        par.blocksize = 20;
        par.stepsize  = 17;
        par.maxIter = 50;
        par.tol = 1e-6;
    end
    Re_hsi = LLRGTV(noisy_hsi, par);
elseif strcmp(methodname, 'FastHyDe')
    noise_type = 'additive';
    p_subspace = 3; %Dimension of the subspace
    iid = 1;
    Re_hsi = FastHyDe(noisy_hsi,  noise_type, iid, p_subspace);
elseif strcmp(methodname, 'GLF')
    noise_type = 'additive';
    p_subspace = 10; %Dimension of the subspace
    Re_hsi = GLF_denoiser(noisy_hsi,  p_subspace,  noise_type) ;
else
    error('Error: no matched method');
end

Re_hsi(Re_hsi>1) = 1;
Re_hsi(Re_hsi<0) = 0;
time = toc;
end
