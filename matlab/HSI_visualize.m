function [vis_img, R_hsi] = HSI_visualize( datadir, resdir, visdir, fn, method, band, leftup, hw, pos, amp_f, translate )
%ECCV_VISUALIZE visualize hsi (groundtruth when strcmp(method,'gt'))
    if ~exist('amp_f', 'var')
        amp_f = ones(size(pos))*6;
    end
    if ~exist('translate', 'var')
        translate = method;
    end
    if ~exist(visdir, 'dir')
        mkdir(visdir);
    end
    disp(['visualize ' method]);     
    filepath = fullfile(datadir, fn);
    mat = load(filepath); % contain (input, gt, sigma)        
    if isfield(mat, 'gt')
        hsi = mat.gt;
    else
        hsi = mat.hsi;
    end
%     hsi = hsi * 0.6;
    
    [~, imgname] = fileparts(fn);
    imgdir = fullfile(resdir, imgname);
    savedir = fullfile(visdir, imgname);
    savepath = fullfile(savedir, [translate '.png']);
    if ~exist(savedir, 'dir')
        mkdir(savedir);
    end
    
    if ~strcmp(method, 'gt') 
        load(fullfile(imgdir, method)); % load R_hsi
    else
        R_hsi = hsi;
    end
    
%     spectrum = R_hsi(60,60,:);
%     plot(spectrum(:), 'DisplayName',method);    
%     hold on
%     legend('show')
%     return 
    
    img = hsi(:,:,band);
%     img = R_hsi(:,:,band); % for real
    maxI = max(img(:));
    minI = min(img(:));
%     disp([maxI, minI]);
    
%     rightbottom = leftup + hw;
    vis_img = (R_hsi(:,:,band)-minI)/(maxI-minI);        
    
%     vis_img = imadjust(vis_img);
    [y, x] = size(vis_img);
    yn = y + 48; xn = x + 48;
    new_img = ones(yn, xn);

    startx = max(floor(xn/2)-floor(x/2), 1);
    starty = floor(yn/2)-floor(y/2);
    new_img(starty:starty+y-1,startx:startx+x-1) = vis_img;
    
    if length(pos) == 1
        new_img = vis_img;
    end
    
    new_img = ShowEnlargedRectangle(new_img, leftup{1}, leftup{1}+hw{1}, amp_f(1), pos(1), 2, [255,0,0]);
%     leftup = [166, 266]; % k = 32
%     leftup = [150, 320]; % k = 6
%     leftup = [55 45]; % for urban
    if length(pos) > 1
        new_img = ShowEnlargedRectangle(new_img, leftup{2}, leftup{2}+hw{2}, amp_f(2), pos(2), 2, [0,255,0]);
    end
    figure
    imshow(new_img);
%     h = imagesc(R_hsi(:,:,band)-img, [-0.03 0.03]);
%     colormap jet
    if ~exist(savepath, 'file') 
        imwrite(new_img, savepath);
    else
        disp([method ' has already existed']);  
    end
%     colorbar('Ticks',[-0.05, -0.025, 0, 0.025, 0.05])        
    axis off
%     title(['\fontsize{25} ' translate]);
%     saveas(h, savepath);      
%     export_fig(savepath)
end

