% 

%% 1. Initialization

% load path
global TPF7_path
global TPF7_filename1
global TPF7_filename5


% MRI scan parameters
TF =  11 ;       % turbo factor
R = 1;           % undersampling factor

% algorithm variables
nfminunc_steps = 20;      % number of sliding steps
global tamer_vars
tamer_vars.track_opt = false;
tamer_vars.pxl_sel_method = 'cm';   % type of pixel selection
pad = 0;

for head_ph_slice = [1]
    
% script variables
exp_name = strcat('_TPF7_sli',num2str(head_ph_slice),'_');
    
%% 2. Load algorithm strucs and channel data, find relevant parameters

load('head_ph_NOmt_256_308.mat');
sens = permute(sens(:,:,head_ph_slice,:),[2 1 3 4]);
im_data = permute(im_data(:,:,head_ph_slice,:),[2 1 3 4]);
kdata_nomotion = permute(kdata(:,:,head_ph_slice,:),[2 1 3 4]);
x_sns = squeeze(sum(conj(sens) .* im_data,4) ./ (eps + sum(abs(sens).^2,4) ));

load('head_ph_mt_256_308.mat','im_data','kdata');
im_data = permute(im_data(:,:,head_ph_slice,:),[2 1 3 4]);
kdata = permute(kdata(:,:,head_ph_slice,:),[2 1 3 4]);

[nlin,ncol,nsli_0,ncha] = size(kdata);
sps = floor(nlin/(R*TF)); tls = sps * nsli_0;

c1 = 15; c2 = 15; c3 = 40; c4 = 40;

%% Create fit function inputs

% pad data as needed based on z-motion
nsli_p = nsli_0 + 2*pad;

% change how mask is done so that it's done for a sli column
msk2 = sum(sum(sens,3),4)~=0;
msk = repmat(msk2,[1 1 nsli_p]);
msk(:,:,:) = 1;
npxl = nlin*ncol*nsli_p;
kdata = cat(3, zeros(nlin,ncol,pad,ncha),kdata,zeros(nlin,ncol,pad,ncha));
im_data = cat(3, zeros(nlin,ncol,pad,ncha),im_data,zeros(nlin,ncol,pad,ncha));
sens = cat(3, repmat(sens(:,:,1,:),[1 1 pad]),sens,repmat(sens(:,:,end,:),[1 1 pad]));

msk_slc = zeros(nlin,ncol,nsli_p);
for cslc = 1:nsli_p
    msk_slc(:,:,cslc) = msk2;
end
nz_pxls = find(msk_slc(:));
km = kdata;

% create U, undersampling matric
U = zeros(size(sens));
U(1:R:end,:,:,:) = 1;

% load tse_traj, sequence kspace trajectory
tse_traj = zeros(tls,TF+1); % "+1" to have first column the slice index
load('tse_traj_151109.mat')
slc_indx = repmat(1:nsli_0, [sps, 1]);
tse_traj(:,1) = slc_indx(:);
tse_traj(:,2:end) = turbo_indx;
 
% find SENSE recon without TAMER
x_km_sns = squeeze(sum(conj(sens) .* im_data,4) ./ (eps + sum(abs(sens).^2,4) ));

% create dM_in_indicies to only optimize
% some of the variables in Mm. Can be used to set intial motion to zero,
% and to restrict to in-plane motion
dM_in_matrix = ones(tls,6);
% dM_in_matrix(1,:) = 0; 
dM_in_matrix(:,3:6) = 0;
dM_in_indices = find(dM_in_matrix);

% initialize kfilter to calculate fit
kfilter = ones(size(km));

Mz = zeros(tls,6);
dM_z = zeros(numel(dM_in_indices),1);

rI_off = [-116,0];

%% 3. Find no motion image and motion corrupted image

[ fit_m, xm, pcg_out] = mt_fit_fcn_v9p( dM_z, dM_in_indices, Mz, sens, kdata_nomotion, ...
    tse_traj, U , nz_pxls , nz_pxls, [], [], kfilter, rI_off);
[ fit_nm, xnm, ~] = mt_fit_fcn_v9p( dM_z, dM_in_indices, Mz, sens, km, ...
    tse_traj, U , nz_pxls , nz_pxls, [], [], kfilter, rI_off);

% mosaic(xm(c1:end-c2,c3:end-c4),1,1,1,'x no motion',[0 max(abs(xm(:)))])
% mosaic(xnm(c1:end-c2,c3:end-c4),1,1,2,'x with motion (no moco)',[0 max(abs(xnm(:)))])

%% 4. PIXEL SELECTION & SHIFT INDEX CREATION     %%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% IMPORTANT!!! - when creating the offsets, they are in terms of "x" and
%%% "y" shifts, not rows and cols, because they are eventually inputs to
%%% imtranslate, which does x and y shifts, not lin and col shift (i.e. run
%%% "m=eye(3);imtranslate(m,[1,0])" for demo, also note the translation 
%%% does not wrap
for foldthiscode = 1

%%%   Method 1. Correlation Matrix
if strcmp(tamer_vars.pxl_sel_method,'cm')
    
    % pixel selection variables
    tar_thre = 1e-2; % ROI reduction factor
    disk_r = 5;      % radius of disk when finding target set
    
    nz_pslc = length(nz_pxls) / nsli_p;
    rtpix_single_sli = find(nz_pxls == ncol*(nlin/2-1) + ncol/2);
    rtpix_all_sli = rtpix_single_sli + (0:nsli_p-1) * nz_pslc;
    rtpix_center_of_vol = rtpix_all_sli(round(end/2));
    rtpix_input = rtpix_center_of_vol; % input root pixel
    
    % use same random motion each time to initialize mask
    rng('default')
    rng(1)
    Mrand = .75*randn(tls,6);
    [ tar_pxls, tar_pxls_comb1, Aprj_comb1] = find_tar_pxls_v6p1( dM_z, ...
        dM_in_indices, Mrand, sens, tse_traj, U , nz_pxls , rtpix_input,...
        pad, nz_pxls, tar_thre, disk_r);
        
    % Create the horizontal shift indices
%     ncol_shift = ceil(ncol / (2 * disk_r));
%     if mod(ncol_shift,2) == 0, ncol_shift = ncol_shift - 1; end
%     citer_vals = zeros(ncol_shift, 1);
%     citer_vals(3:2:end) = (2 * disk_r) * (1:floor(ncol_shift / 2) );
%     citer_vals(2:2:end) = -citer_vals(3:2:end);
    
    %%%%%%% specific for head phantom data to move pixels more efficiently
    ncol_shift = ceil((ncol/2) / (2 * disk_r));
    if mod(ncol_shift,2) == 0, ncol_shift = ncol_shift - 1; end
    citer_vals = zeros(ncol_shift, 1);
    citer_vals(3:2:end) = (2 * disk_r) * (1:floor(ncol_shift / 2) );
    citer_vals(2:2:end) = -citer_vals(3:2:end);
    
    % Create the vertical/extra horizontal shift indices
    citer_offsets = [0 0;
        0 2*disk_r;
        0 -2*disk_r;
        disk_r 0;
        disk_r disk_r;
        disk_r -disk_r];
          
    % create exp string for pixel selection method
    pxl_sel_str = num2str(tar_thre); pxl_sel_str = pxl_sel_str(3:end);
    pxl_sel_str = strcat('pt',pxl_sel_str,'th');
    
%%%   Method 2. Correlation Matrix Rotated
elseif strcmp(tamer_vars.pxl_sel_method,'cmRot')
    
    % pixel selection variables
    tar_thre = 1e-2; % ROI reduction factor
    disk_r = 5;      % radius of disk when finding target set
    
    nz_pslc = length(nz_pxls) / nsli_p;
    rtpix_single_sli = find(nz_pxls == ncol*(nlin/2-1) + ncol/2);
    rtpix_all_sli = rtpix_single_sli + (0:nsli_p-1) * nz_pslc;
    rtpix_center_of_vol = rtpix_all_sli(round(end/2));
    rtpix_input = rtpix_center_of_vol; % input root pixel
    
    % use same random motion each time to initialize mask
    rng('default')
    rng(1)
    Mrand = .75*randn(tls,6);
    [ tar_pxls, tar_pxls_comb1, Aprj_comb1] = find_tar_pxls_v6p1( dM_z, ...
        dM_in_indices, Mrand, sens, tse_traj, U , nz_pxls , rtpix_input,...
        pad, nz_pxls, tar_thre, disk_r);
        
        
    rot_mask = zeros(nlin,ncol); rot_mask(tar_pxls) = 1;
    rot_mask = imrotate(rot_mask,90);
    tar_pxls = find(rot_mask);
    
    % Create the horizontal shift indices
    ncol_shift = ceil(ncol / (2 * disk_r));
    if mod(ncol_shift,2) == 0, ncol_shift = ncol_shift - 1; end
    citer_vals = zeros(ncol_shift, 1);
    citer_vals(3:2:end) = (2 * disk_r) * (1:floor(ncol_shift / 2) );
    citer_vals(2:2:end) = -citer_vals(3:2:end);
    
    % Create the vertical/extra horizontal shift indices
    citer_offsets = [0 0;
        0 2*disk_r;
        0 -2*disk_r;
        disk_r 0;
        disk_r disk_r;
        disk_r -disk_r];
          
    % create exp string for pixel selection method
    pxl_sel_str = num2str(tar_thre); pxl_sel_str = pxl_sel_str(3:end);
    pxl_sel_str = strcat('pt',pxl_sel_str,'th_rot');
    
    %%%   Method 3. Uniform - changed to circles 3/24
elseif strcmp(tamer_vars.pxl_sel_method,'uni')
    disk_r = 5;
    per_pxl = .033;
    
    im_test = zeros(nlin,ncol); im_test(round(nlin/2),round(ncol/2)) = 1;
    im_test_dil = imdilate(im_test, strel('disk', disk_r));
    npxl_per_box = nnz(im_test_dil);

    ntar_pxls = round(per_pxl*numel(nz_pxls));
    ncir = round(ntar_pxls / npxl_per_box);
    
    % gernal but not exact method
%     nbox_cir_dir = round(sqrt(ncir));
%     cir_spacing = round(nlin/nbox_cir_dir);
%     cir_center_indices = (round(cir_spacing/2):cir_spacing:nlin)';
%     im_cir_centers = zeros(nlin,ncol);
%     im_cir_centers(cir_center_indices,cir_center_indices) = 1;
%     im_cir = imdilate(im_cir_centers, strel('disk', cir_r));
    
    
    % specifically for 8 circles in 3-2-3 order - used pre 3/27
%     cir_spacing3 = round(nlin/3);
%     cir_spacing2 = round(nlin/2);
%     cir_sp3_vec = (round(cir_spacing3/2):cir_spacing3:nlin)';
%     cir_sp2_vec = (round(cir_spacing2/2):cir_spacing2:nlin)';  
%     cir_row_ind = [repmat(cir_sp3_vec(1),1,3),...
%         repmat(cir_sp3_vec(2),1,2),repmat(cir_sp3_vec(3),1,3)];
%     cir_col_ind = [cir_sp3_vec; cir_sp2_vec; cir_sp3_vec];
%     im_cir_centers = zeros(nlin,ncol);
%     im_cir_centers(sub2ind([nlin, ncol],cir_row_ind',cir_col_ind)) = 1;
%     im_cir = imdilate(im_cir_centers, strel('disk', disk_r));

    % specifically for 7 circles in 2-3-2 column order - used from 3/27 on
    sp_3 = round(nlin/3);
    cir_sp3_v = (round(sp_3/2):sp_3:nlin)';
    cir_sp2_v = [sp_3; 2*sp_3];
    
    cir_row_ind = [cir_sp3_v(1),cir_sp2_v(1),cir_sp2_v(1),...
        cir_sp3_v(2),cir_sp2_v(2),cir_sp2_v(2),cir_sp3_v(3)];
    cir_col_ind = [cir_sp3_v(2),cir_sp3_v(1),cir_sp3_v(3),...
        cir_sp3_v(2),cir_sp3_v(1),cir_sp3_v(3),...
        cir_sp3_v(2)];
    im_cir_centers = zeros(nlin,ncol);
    im_cir_centers(sub2ind([nlin, ncol],cir_row_ind,cir_col_ind)) = 1;    
    im_cir_centers2 = imdilate(im_cir_centers,strel('arbitrary',[0 0 0; 0 1 1; 0 0 0]));
    im_cir = imdilate(im_cir_centers2, strel('disk', disk_r));
    
    tar_pxls = find(im_cir);
    
    % Create the horizontal shift indices
    ncol_shift = ceil(ncol / (2 * disk_r));
    if mod(ncol_shift,2) == 0, ncol_shift = ncol_shift - 1; end
    citer_vals = zeros(ncol_shift, 1);
    citer_vals(3:2:end) = (2 * (disk_r-1)) * (1:floor(ncol_shift / 2) );
    citer_vals(2:2:end) = -citer_vals(3:2:end);

    % Create the vertical/extra horizontal shift indices
    citer_offsets = [zeros(9,1), [0,1,-1,2,-2,3,-3,4,-4]'*disk_r];
        
    % create exp string for pixel selection method
    pxl_sel_str = strcat('uni',num2str(100*per_pxl),'p');
    pxl_sel_str = strrep(pxl_sel_str,'.','pt');

%%%   Method Random Circs - created 3/28
elseif strcmp(tamer_vars.pxl_sel_method,'randcir')
    disk_r = 5;
    per_pxl = .033;
    
%     im_test = zeros(nlin,ncol); im_test(round(nlin/2),round(ncol/2)) = 1;
%     im_test_dil = imdilate(im_test, strel('disk', disk_r));
%     npxl_per_box = nnz(im_test_dil);
% 
%     ntar_pxls = round(per_pxl*numel(nz_pxls));
%     ncir = round(ntar_pxls / npxl_per_box);

    % specifically for 7 circles 
    
    % use same random locations to initialize mask
    rng('default')
    rng(3)
    rand_indices = randperm(numel(nz_pxls));
    cir_row_ind = nz_pxls(rand_indices(1:7));

    im_cir_centers = zeros(nlin,ncol);
    im_cir_centers(cir_row_ind) = 1;    
    im_cir_centers2 = imdilate(im_cir_centers,strel('arbitrary',[0 0 0; 0 1 1; 0 0 0]));
    im_cir = imdilate(im_cir_centers2, strel('disk', disk_r));
    
    tar_pxls = find(im_cir);
    
    % Create the horizontal shift indices
    ncol_shift = ceil(ncol / (2 * disk_r));
    if mod(ncol_shift,2) == 0, ncol_shift = ncol_shift - 1; end
    citer_vals = zeros(ncol_shift, 1);
    citer_vals(3:2:end) = (2 * (disk_r-1)) * (1:floor(ncol_shift / 2) );
    citer_vals(2:2:end) = -citer_vals(3:2:end);

    % Create the vertical/extra horizontal shift indices
    citer_offsets = [zeros(13,1), [0,1,-1,2,-2,3,-3,4,-4,5,-5,6,-6]'*disk_r];
        
    % create exp string for pixel selection method
    pxl_sel_str = strcat('randcir',num2str(100*per_pxl),'p');
    pxl_sel_str = strrep(pxl_sel_str,'.','pt');
    
elseif strcmp(tamer_vars.pxl_sel_method,'bigROI')
    per_pxl = .033;
    
    ntar_pxls = round(per_pxl*numel(nz_pxls));
    disk_r = round(sqrt(ntar_pxls/pi));
    im_cir_center = zeros(nlin,ncol);
    im_cir_center(round(nlin/2),round(ncol/2)) = 1;
    im_cir = imdilate(im_cir_center, strel('disk', disk_r));
    
    tar_pxls = find(im_cir);
    
    % Create the horizontal shift indices
    ncol_shift = ceil((ncol/2) / (2 * disk_r)) + 2;
    if mod(ncol_shift,2) == 0, ncol_shift = ncol_shift - 1; end
    citer_vals = zeros(ncol_shift, 1);
    citer_vals(3:2:end) = (2 * (disk_r-1)) * (1:floor(ncol_shift / 2) );
    citer_vals(2:2:end) = -citer_vals(3:2:end);
    citer_vals = [citer_vals; citer_vals(end-1)+citer_vals(2)]; % add extra col for bigROI

    % Create the vertical/extra horizontal shift indices
    citer_offsets = [zeros(9,1), [0,1,-1,2,-2,3,-3,4,-4]'*disk_r];
    
    % create exp string for pixel selection method
    pxl_sel_str = strcat('bigROI',num2str(100*per_pxl),'p');
    pxl_sel_str = strrep(pxl_sel_str,'.','pt');

end

% Combine horizontal and vertical shifts into one vector
citer_vals_repmat = repmat(cat(2,citer_vals,zeros(size(citer_vals))),size(citer_offsets,1),1);
citer_vals_offsets = kron(citer_offsets,ones(numel(citer_vals),1));
tamer_vars.citer_vals_all = citer_vals_repmat + citer_vals_offsets;
end

%% SET COIL SUBSET FOR TAMER OPTIMIZATION
cha_subset = 1:4;   %%%%% WHERE SUBSET DEFINED, not really used for head ph
sens_sub = sens(:,:,:,cha_subset);
kdata_nomotion_sub = kdata_nomotion(:,:,:,cha_subset);
km_sub = km(:,:,:,cha_subset);
U_sub = U(:,:,:,cha_subset);
kfilter_sub = kfilter(:,:,:,cha_subset);

% select full coil set or subset of channels for tamer (tm)
run_subset = false;
if run_subset
    sens_tm = sens_sub;
    km_tm = km_sub;
    U_tm = U_sub;
    kfilter_tm = kfilter_sub;
else
    sens_tm = sens;
    km_tm = km;
    U_tm = U;
    kfilter_tm = kfilter;
end


%% 5. setup search

exp_str = strcat(datestr(now,'yyyy-mm-dd'),exp_name,...
       pxl_sel_str,'_R',num2str(R),'_TF',num2str(TF),'_');
exp_path = exp_str; exp_path(end) = '/';
while exist(exp_path,'dir')
    exp_path = strcat(exp_path(1:end-1),'i/');
    exp_str = strcat(exp_str(1:end-1),'i_');
end
mkdir(exp_path);

% tamer algorithm vars

tamer_vars.cup = 0;   % how many times the objective has been updated
tamer_vars.call = 0;  % total number of function calls

% optimization tracking variables
tamer_vars.nfixed_objfnc_calls = 0;
tamer_vars.nobjfnc_calls = 0;
tamer_vars.ngrad_calc = 0;
tamer_vars.ntotal_pcg_steps = 0;
tamer_vars.pcg_steps = [];
tamer_vars.nmotion_traj_attmpt = 0;
tamer_vars.fit_vec = [];
tamer_vars.tar_pxl_all = [];
tamer_vars.per_tar_v = [];

save_updates = false;
hardcode_search_options = true;

%%%%% changes 6/6/17 and 6/7/17
Mn_prev = Mz;
tamer_vars.fit_init = fit_nm;
tamer_vars.xprev_best = xnm;


% set fminunc options
if hardcode_search_options
    % max_feval = length(tamer_vars.citer_vals) + 2;
    max_feval = 13; % hardcoded until I can find a better method to compare
                    % the different citer_vals for each pixel method
    max_iter = 2 * max_feval;
    
    tamer_opt = optimoptions(@fminunc, 'Algorithm','quasi-newton',...
        'MaxIter',max_iter,'Display','iter','SpecifyObjectiveGradient',true, ...
        'MaxFunctionEvaluations', max_feval);
else
    tamer_opt = optimoptions(@fminunc, 'Algorithm','quasi-newton',...
        'Display','iter','SpecifyObjectiveGradient',true);
end



%% Run Optimization

tamer_start = tic;

disp('  '); disp(strcat(exp_str,'1-',num2str(nfminunc_steps))); disp('  ')
for ii = 1:nfminunc_steps
    
    disp('  '); disp(strcat(exp_str,num2str(ii))); disp('  ');
    diary(strcat(exp_path,exp_str,num2str(ii)));
    
    %%%%% 8/31 addition
    tamer_vars.track_opt = true;
    % search across x_targetted and update motion
    [dr_tmp, fit_tmp, exit_fl] = fminunc(@(dM_tmp) mt_fit_fcn_v9p_grad_tarSh( dM_tmp,...
        dM_in_indices, Mn_prev, sens_tm, km_tm, tse_traj, U_tm , tar_pxls, ...
        nz_pxls, kfilter_tm, exp_str, exp_path, save_updates, rI_off), ...
        zeros(numel(dM_in_indices),1), tamer_opt);
    %%%%% 8/31 addition
    tamer_vars.track_opt = false;

    Mn_prev(dM_in_indices) = Mn_prev(dM_in_indices) + dr_tmp;
       
    tamer_intermediate_time = toc(tamer_start);
    
    % save search progress
    save(strcat(exp_path,exp_str,num2str(ii),'.mat'), 'Mn_prev','dr_tmp',...
        'fit_tmp','tar_pxls','tamer_vars','tamer_intermediate_time');
    
end

% find final tamer fit all channels
[ fit_tamer, xtamer, ~] = mt_fit_fcn_v9p( dM_z, dM_in_indices, Mn_prev, sens, km, ...
    tse_traj, U , nz_pxls , nz_pxls, [], [], kfilter, rI_off);

tamer_end = toc(tamer_start);


% save workspace
if head_ph_slice == 1
    TPF7_filename1 = strcat(TPF7_path,'/',exp_path,exp_str,'end_wrksp.mat');
    save(TPF7_filename1)
elseif head_ph_slice == 5
    TPF7_filename5 = strcat(TPF7_path,'/',exp_path,exp_str,'end_wrksp.mat');
    save(TPF7_filename5)
end

end




