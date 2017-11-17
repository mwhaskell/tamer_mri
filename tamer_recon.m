% tamer_recon.m

% Example code for Figure 8 of "TArgeted Motion Estimation and Reduction
% (TAMER): Data Consistency Based Motion Mitigation for MRI using a
% Reduced Model Joint Optimization" by Melissa W. Haskell,
% Stephen F. Cauley, and Lawrence L. Wald

% Last updated: Nov 16, 2017

%% 1. Initialization

% add paths
addpath('./tamer_data');
addpath('./funcs_tamer');

% algorithm variables
nsteps = 20;        % -number of gradient reset steps
sli = 1;            % -select slice to correct
pad = 0;            % -amount of zero padding in slice direction
global tamer_vars
exp_name = '_tamer_demo_';  % -experiment name
show_figs = true;   % -set whether or not to view figures

c1 = 15; c2 = 15; c3 = 40; c4 = 40; % -cropping parameters for plotting

%% 2. Load and zeropad data, find relevant parameters

% load data
if sli == 1, load('data_sli1.mat');
elseif sli == 2, load('data_sli2.mat'); end

% zero pad
[nlin,ncol,~,ncha] = size(kdata);
kdata = cat(3, zeros(nlin,ncol,pad,ncha),kdata,zeros(nlin,ncol,pad,ncha));
sens = cat(3, repmat(sens(:,:,1,:),[1 1 pad]),sens,repmat(sens(:,:,end,:),[1 1 pad]));
nsli = size(kdata,3);

sps = size(turbo_indx,1);       % -sps = "shots per slice"
TF = size(turbo_indx,2);        % -TF = "turbo factor"
tls = sps * nsli;               % -tls = "total shots"
R = nlin / (sps*TF);            % -R = undersampling factor

%% 3. Create additional objective function inputs

% image space mask
msk = sum(sum(sens,3),4)~=0;
msk = repmat(msk,[1 1 nsli]);
msk_vxls = find(msk);           % -msk_vxls = "mask voxel indices"

% create U, undersampling matrix
U = zeros(size(sens));
U(1:R:end,:,:,:) = 1;

% create tse_traj, sequence k-space trajectory
tse_traj = zeros(tls,TF+1); % "+1" to have first column the slice index
slc_indx = repmat(1:nsli, [sps, 1]);
tse_traj(:,1) = slc_indx(:);
tse_traj(:,2:end) = turbo_indx;

% create mt2corr to select which motion variables to optimize
mt2corr = ones(tls,6);  % -mt2corr = "motion parameters to correct for"
mt2corr(:,3:6) = 0;     %   this example only correct in plane translation,
                        %   can remove this line to correct all six rigid
                        %   body motion parameters
mt2corr = find(mt2corr);

% initialize kfilter to calculate fit
kfilt = ones(size(kdata));      % -kfilt = "k-space data consistency
                                %    weighing filter", not used in this
                                %    example, so set to all ones

% create motion variables
theta = zeros(tls,6);               % -theta represented in a Nshot x 6
                                    %    matrix here
dTheta_0 = zeros(numel(mt2corr),1); % -dTheta_0 = "Delta theta equal to 0"

rI_off = [0,0];     % -rotation offset for in-plane rotation ("rI")

%% 3. Find initial motion corrupted image

% find the initial image by calling the TAMER objective function. 
%    "tamer_obj_func_par" uses the parallel threaded objective function
[ fit_0, x0, ~, ks] = tamer_obj_func_par( dTheta_0, mt2corr, theta, sens, kdata, ...
    tse_traj, U , msk_vxls , msk_vxls, [], [], kfilt, rI_off);
if show_figs, mosaic(x0(c1:end-c2,c3:end-c4),1,1,1,'x0',[0 max(abs(x0(:)))]); end


%% 4. Voxel Selection

% voxel selection variables
tar_thre = 1e-2; % ROI reduction factor
disk_r = 5;      % radius of disk when finding target set
rtvox_indx = find(msk_vxls == (nlin/2)*ncol + ncol/2); % "root" voxel index

% -use same random motion each time for mask intializing (done for code
%   repeatability, but one could use different motion each time)
rng('default'); rng(1)
thetaRand = .75*randn(tls,6);
[ tar_vxls, ~, ~] = find_tar_pxls_v6p1( dTheta_0, ...
    mt2corr, thetaRand, sens, tse_traj, U , msk_vxls , rtvox_indx,...
    pad, msk_vxls, tar_thre, disk_r);

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

% Combine horizontal and vertical shifts into one vector
citer_vals_repmat = repmat(cat(2,citer_vals,zeros(size(citer_vals))),size(citer_offsets,1),1);
citer_vals_offsets = kron(citer_offsets,ones(numel(citer_vals),1));
tamer_vars.citer_vals_all = citer_vals_repmat + citer_vals_offsets;


%% 5. Channel grouping 
imerr = abs(ifft2c(kdata-ks));
imerr_corr = zeros(ncha);
for ii = 1:ncha
    for jj = ii:ncha
        imerr_corr(ii,jj) = corr2(imerr(:,:,ii),imerr(:,:,jj));
        imerr_corr(jj,ii) = imerr_corr(ii,jj);
    end
end
if show_figs, 
    figure(2); imagesc(imerr_corr); 
    colormap gray; title('data consistency error correlation')
end

cha_group = 1:ncha;   % here only a single cluster, so use all channels
sens = sens(:,:,:,cha_group);
kdata = kdata(:,:,:,cha_group);
U = U(:,:,:,cha_group);
kfilt = kfilt(:,:,:,cha_group);


%% 6. Run Joint Optimization

exp_str = strcat(datestr(now,'yyyy-mm-dd'),exp_name);
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
% tamer_vars.nfixed_objfnc_calls = 0;
% tamer_vars.nobjfnc_calls = 0;
% tamer_vars.ngrad_calc = 0;
% tamer_vars.ntotal_pcg_steps = 0;
% tamer_vars.pcg_steps = [];
% tamer_vars.nmotion_traj_attmpt = 0;
% tamer_vars.fit_vec = [];
% tamer_vars.tar_pxl_all = [];
% tamer_vars.per_tar_v = [];

% save_updates = false;
% hardcode_search_options = true;

%%%%% changes 6/6/17 and 6/7/17
theta_prev = theta;
tamer_vars.fit_init = fit_0;
tamer_vars.xprev_best = x0;


% max_feval = length(tamer_vars.citer_vals) + 2;
max_feval = 13; % hardcoded until I can find a better method to compare
% the different citer_vals for each pixel method
max_iter = 2 * max_feval;

tamer_opt = optimoptions(@fminunc, 'Algorithm','quasi-newton',...
    'MaxIter',max_iter,'Display','iter','SpecifyObjectiveGradient',true, ...
    'MaxFunctionEvaluations', max_feval);

tamer_start = tic;

disp('  '); disp(strcat(exp_str,'1-',num2str(nsteps))); disp('  ')
for ii = 1:nsteps
    
    disp('  '); disp(strcat(exp_str,num2str(ii))); disp('  ');
    
    % search across x_targetted and update motion
    [dTheta_tmp, fit_tmp, exit_fl] = fminunc(@(dM_tmp) tamer_obj_func_wGrad( dM_tmp,...
        mt2corr, theta_prev, sens, kdata, tse_traj, U , tar_vxls, ...
        msk_vxls, kfilt, exp_str, exp_path, save_updates, rI_off), ...
        zeros(numel(mt2corr),1), tamer_opt);

    
    theta_prev(mt2corr) = theta_prev(mt2corr) + dTheta_tmp;
    
    tamer_intermediate_time = toc(tamer_start);
    
    % save search progress
    save(strcat(exp_path,exp_str,num2str(ii),'.mat'), 'theta_prev','dTheta_tmp',...
        'fit_tmp','tar_vxls','tamer_vars','tamer_intermediate_time');
    
end

%% 7. Final full volume solve
[ fit_tamer, xtamer, ~] = mt_fit_fcn_v9p( dTheta_0, mt2corr, theta_prev, sens, kdata, ...
    tse_traj, U , msk_vxls , msk_vxls, [], [], kfilt, rI_off);

tamer_end = toc(tamer_start);
tamer_end_min = tamer_end / 60;
disp(strcat('TAMER run time: ',tamer_end_min,' min'))

% save end workspace
save(strcat(exp_path,exp_str,num2str(ii),'_end_wrksp.mat'));







