% 

global TPF7_filename1
global TPF7_filename5

%% slice 1
load(TPF7_filename1,'xtamer','xm','xnm','Mn_prev')
final_im_1 = xtamer;
xnmoco_sli1 = xnm;
xm1_in = xm;

[ rerr1, xm1] = reg_img ([0,0], xm1_in, xtamer);
err1 = final_im_1 - xm1;
im_tamer_fit_1_0 = norm( err1(:) )/norm(xm1(:));

im_tamer_fit_min = im_tamer_fit_1_0;
shift_min1 = [0,0];
xm1_min = xm1_in;

% register no motion image to tamer im
rsh_vals = -4.05:.05:1.05; % -3.85
csh_vals = -3.05:.05:3.05; % -0.15
for ii = 1:numel(rsh_vals)
    for jj = 1:numel(csh_vals)
        
        rsh = rsh_vals(ii); csh = csh_vals(jj);
        
        [ rerr1, xm1_loop] = reg_img ([rsh,csh], xm1_in, final_im_1);
        err1 = final_im_1 - xm1_loop;
%         [ rerr1, xm1_loop] = reg_img ([rsh,csh], xm1_in, xnmoco_sli1);
%         err1 = xnmoco_sli1 - xm1_loop;
        im_tamer_fit_1_loop = norm( err1(:) )/norm(xm1_loop(:));
        
        if im_tamer_fit_1_loop < im_tamer_fit_min
            im_tamer_fit_min = im_tamer_fit_1_loop;
            shift_min1 = [rsh,csh];
            xm1_min = xm1_loop;
        end
    end
end

xm1 = xm1_min;
err1 = final_im_1 - xm1;
nmErr1 = xnmoco_sli1 - xm1;
Msh_sli1 = Mn_prev;
clear xtamer xm xnm Mn_prev


%% slice 5
load(TPF7_filename5,'xtamer','xm','xnm','Mn_prev')
final_im_5 = xtamer;
xnmoco_sli5 = xnm;
xm5_in = xm;


[ rerr5, xm5] = reg_img ([0,0], xm5_in, final_im_5);
err5 = final_im_5 - xm5;
im_tamer_fit_5_0 = norm( err5(:) )/norm(xm5(:));

im_tamer_fit_min = im_tamer_fit_5_0;
shift_min5 = [0,0];
xm5_min = xm5_in;

% register no motion image to tamer im
rsh_vals = -4.5:.05:1.05;  % -3.9
csh_vals = -3.05:.05:3.05; % -.10
for ii = 1:numel(rsh_vals)
    for jj = 1:numel(csh_vals)
        
        rsh = rsh_vals(ii); csh = csh_vals(jj);
        
        [ rerr5, xm5_loop] = reg_img ([rsh,csh], xm5_in, final_im_5);
        err5 = final_im_5 - xm5_loop;
        im_tamer_fit_5_loop = norm( err5(:) )/norm(xm5_loop(:));
        
        if im_tamer_fit_5_loop < im_tamer_fit_min
            im_tamer_fit_min = im_tamer_fit_5_loop;
            shift_min5 = [rsh,csh];
            xm5_min = xm5_loop;
        end
    end
end

xm5 = xm5_min;
err5 = final_im_5 - xm5;
nmErr5 = xnmoco_sli5 - xm5;
Msh_sli5 = Mn_prev;
clear xtamer xm xnm Mn_prev

%% Calculate Error
im_tamer_fit_1 = norm( err1(:) )/norm(xm1(:));
im_tamer_fit_5 = norm( err5(:) )/norm(xm5(:));
im_nm_fit_1 = norm( nmErr1(:) )/norm(xm1(:));
im_nm_fit_5 = norm( nmErr5(:) )/norm(xm5(:));

%% Print Error Values
fprintf('Sli 1 TAMER RMSE: %.3f, Sli 1 motion corrupted RMSE: %.3f\n',...
    im_tamer_fit_1,im_nm_fit_1);
fprintf('Sli 2 TAMER RMSE: %.3f, Sli 2 motion corrupted RMSE: %.3f\n',...
    im_tamer_fit_5,im_nm_fit_5);


%% Plot Images
s=4;
lc = 78; rc = 68; tc = 30; bc = 30;
images = cat(3,xm1,xnmoco_sli1,s*nmErr1,final_im_1,s*err1,...
               xm5,xnmoco_sli5,s*nmErr5,final_im_5,s*err5);
images = images(tc:end-bc,lc:end-rc,:);
mosaic(images, 2,5, 11,'',[0 max(abs(xm5(:)))/1.5])


%% Create motion plots
% (formatting changes done by hand using plot tools)
figure('units','normalized','outerposition',[0 0 1 1])
hold on;
plot(Msh_sli1(:,2),'Color',[.5 .5 .5],'LineWidth',10)
plot(Msh_sli1(:,1),'k-.','LineWidth',10)
box on;
set(gcf,'color','w');
set(gca,'yaxislocation','right','LineWidth',4)
axis([0.5 29 -4.5 1.9])
set(gca,'FontSize',90,'FontName','Arial')


figure('units','normalized','outerposition',[0 0 1 1])
hold on;
plot(Msh_sli5(:,2),'Color',[.5 .5 .5],'LineWidth',10)
plot(Msh_sli5(:,1),'k-.','LineWidth',10)
box on;
set(gcf,'color','w');
set(gca,'yaxislocation','right','LineWidth',4)
axis([0.5 29 -4.9 1.9])
set(gca,'FontSize',90,'FontName','Arial')