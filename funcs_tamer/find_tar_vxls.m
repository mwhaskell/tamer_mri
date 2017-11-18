function [ tar_vxls_sm, tar_vxls_indx, EHE_v] = find_tar_vxls( dTheta_in,...
    mt2corr, theta_prev, sens,  tse_traj, U , msk_vxls, root_indx, pad, ...
    tar_cutoff, disk_r)

% Find target voxels
%
% INPUTS:
%
%   dTheta_in: delta theta (change in motion trajectory vector)
%   mt2corr: specifies which motion parameters to correct
%   theta_prev: previous estimate of theta
%   sens: sensitivity maps
%   tse_traj: TSE shot trajectory (sampling trajectory)
%   U: undersampling matrix
%   msk_vxls: voxels of mask
%   root_indx: index of root voxel
%   pad: padding in slice direction
%   tar_cutoff: cutoff to include a voxel in the target set
%   disk_r: radius of disk used to smooth target voxel
%
% OUTPUTS:
%
%   tar_vxls_sm: target voxels smoothed
%   tar_vxls_indx: indices of target voxels before smoothing
%   EHE_v: row vector from E*H for this root voxel

%%                             Precomputations                           %%

%%% Currently hardcoded values
lambda = 0;
[nlin,ncol,nsli_0,~] = size(U);

nsli_p = nsli_0 + 2*pad;

% reshape motion vectors
dTheta_in_all = zeros(numel(theta_prev),1);
dTheta_in_all(mt2corr) = dTheta_in;
dTheta_in_all_mtx = reshape(dTheta_in_all, size(theta_prev));
theta = theta_prev + dTheta_in_all_mtx;

euc = zeros(length(msk_vxls), 1);
euc(root_indx) = 1;

EHE_v = EHE(euc,U,sens,tse_traj,theta,lambda,msk_vxls);

tar_vxls_indx = find(abs(EHE_v(:)) > max(abs(EHE_v)) * tar_cutoff);

%  make it z uniform and crop along column dimension
img_comb = zeros(nlin,ncol,nsli_p); img_comb(msk_vxls(tar_vxls_indx)) = 1;
img_comb = repmat(sum(img_comb, 3), 1, 1, nsli_p);


[~, mindx] = max(sum(img_comb(:,:,1), 1));
img_comb(:,1:mindx-disk_r,:) = 0;
img_comb(:,mindx+disk_r:end,:) = 0;

% create larger version to accomidate movement
img_comb_large = imdilate(img_comb, strel('disk', disk_r));
tar_vxls_sm = find(img_comb_large);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%        E - encoding forward model       %%%%%%%%%%%%%%%%%%%%%%%%%%%
function [kdata] = E(x,U,sens,tse_traj,theta,recon_vxls)

%% precomputations
[nlin, ncol, nsli, ncha] = size(U);

% find R
if numel(U) == numel(find(U)), R = 1;
else R = round(numel(U)/numel(find(U))); end

% find sequence parameters
TF = size(tse_traj,2) - 1;
tls = size(tse_traj,1);
sps = nlin/(R*TF);
pad = ( nsli - tls/sps )/2;

%% begin forward model

% FMx for "forward model x"
FMx = zeros(size(U));
FMx_input_mtx = zeros(TF,ncol,ncha,tls);
kp_vec = linspace(-pi,pi-2*pi*(1/nsli),nsli);
kr_vec = linspace(-pi,pi-2*pi*(1/nlin),nlin); kr_mtx = repmat(kr_vec.',1,ncol);
kc_vec = linspace(-pi,pi-2*pi*(1/ncol),ncol); kc_mtx = repmat(kc_vec,nlin,1);
kspace_2d = cat(3,kr_mtx, kc_mtx);

dx_v = theta(:,1);dy_v = theta(:,2);dz_v = theta(:,3);
yaw_v = theta(:,4); pitch_v = theta(:,5); roll_v = theta(:,6);
tse_traj_mtx = tse_traj(:,2:end);


Cfull2 = zeros(nlin,ncol,tls,ncha);
for t = 1:tls
    tmp_sli = tse_traj(t,1) + pad;
    Cfull2(:,:,t,:) = sens(:,:,tmp_sli,:);
end

p = gcp('nocreate'); % If no pool, do not create new one.
if isempty(p)
    parpool(feature('numCores'));
end
parfor t = 1:tls
    tmp_sli = tse_traj(t,1) + pad;
    tmp_tse_traj = tse_traj_mtx(t,:);
    
    dx = dx_v(t);
    dy = dy_v(t);
    dz = dz_v(t);
    yaw    = yaw_v(t);
    pitch  = pitch_v(t);
    roll   = roll_v(t);
    
    % R
    temp_vol = zeros(nlin,ncol,nsli);
    temp_vol(recon_vxls) = x;
    temp_volR = rot3d(temp_vol,yaw,pitch,roll);

    nz_pxls_rot = find(repmat(sum(abs(temp_volR),3), 1, 1, nsli));
    nz_pxls_shot = union(recon_vxls,nz_pxls_rot);
    nz_im = zeros(nlin,ncol,nsli); nz_im(nz_pxls_shot) = 1;
    nz_im = sum(nz_im,3);
    sli_pxls_shot = find(nz_im);
    
    Rx = reshape(temp_volR(nz_pxls_shot),numel(sli_pxls_shot),nsli);
    
    % Fz
    FzRx = fftshift(fft(ifftshift(Rx,2), nsli, 2) ,2);
    
    % Mz
    p_ph = exp(-1i * kp_vec * dz).';
    MzFzRx = permute(repmat(p_ph,1,numel(sli_pxls_shot)),[2 1]) .* FzRx;
    
    % Fzin
    FzinMzFzRx = fftshift(ifft(ifftshift(MzFzRx, 2), nsli, 2) ,2);
    
    % Uss
    UssFzinMzFzRx = zeros(nlin*ncol,1);
    UssFzinMzFzRx(sli_pxls_shot,:) = FzinMzFzRx(:,tmp_sli);
    UssFzinMzFzRx = reshape(UssFzinMzFzRx,nlin,ncol);
    
    % Fxy
    FxyUssFzinMzFzRx = fftshift(fftshift(fft2(...
        ifftshift(ifftshift(UssFzinMzFzRx,1),2)),1),2);
    
    % Mxy
    % create motion matrix (assume standard cartesian sampling)
    mmtx_sli = cat(3, repmat(dx,nlin,ncol), repmat(dy,nlin,ncol));
    Mxy_sli = exp(-1i * sum(kspace_2d.*mmtx_sli,3) );
    MxyFxyUssFzinMzFzRx= Mxy_sli .* FxyUssFzinMzFzRx;
    
    % Fxyin
    FxyinMxyFxyUssFzinMzFzRx = fftshift(fftshift(ifft2(...
        ifftshift(ifftshift(MxyFxyUssFzinMzFzRx,1),2)),1),2);
    
    % C
    Cx = squeeze(Cfull2(:,:,t,:)) .* repmat(FxyinMxyFxyUssFzinMzFzRx,1,1,ncha);
    
    % Fen
    FenCx = fftshift(fftshift(fft2(...
        ifftshift(ifftshift(Cx,1),2) ) ,1),2);
    
    % Uss
    FMx_input = FenCx(tmp_tse_traj,:,:);
    FMx_input_mtx(:,:,:,t) = FMx_input;
      
end

for t = 1:tls
    tmp_sli = tse_traj(t,1) + pad;
    tmp_tse_traj = tse_traj_mtx(t,:);
    FMx(tmp_tse_traj,:,tmp_sli,:) = FMx_input_mtx(:,:,:,t);
end

kdata = FMx;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%      EH - E Hermitian (E*), reverse model    %%%%%%%%%%%%%%%%%%%
function [imdata] = EH(k,U,sens,tse_traj,theta,recon_vxls)

%% precomputations
[nlin, ncol, nsli, ncha] = size(U);

nz_im = zeros(nlin,ncol,nsli); nz_im(recon_vxls) = 1;
sli_pxls_in = find(nz_im(:,:,round(end/2)));

% find R
if numel(U) == numel(find(U)), R = 1;
else R = round(numel(U)/numel(find(U))); end

% find sequence parameters
TF = size(tse_traj,2) - 1;
tls = size(tse_traj,1);
sps = nlin/(R*TF);
pad = ( nsli - tls/sps )/2;


%% begin reverse model

% prep variables for parellelization
sli_traj = tse_traj(:,1); shot_traj = tse_traj(:,2:end);
dz_v = theta(:,3); yaw_v = theta(:,4); pitch_v = theta(:,5); roll_v = theta(:,6);

%%% Uu* operator
% reorganizes data into format based on shots
Uusk = zeros(nlin,ncol,tls,ncha);
for t = 1:tls
    
    tmp_sli = sli_traj(t) + pad;
    tmp_tse_traj = shot_traj(t,:);
    Uusk(tmp_tse_traj,:,t,:) = k(tmp_tse_traj,:,tmp_sli,:);
    
end

%%% Fen* operator
FensUusk = fftshift(fftshift( ifft2(ifftshift(ifftshift(Uusk,1),2)) ,1),2);

%%% C* operator
CsFensUusk = zeros(nlin,ncol,tls);
for t = 1:tls
    tmp_sli = sli_traj(t) + pad;
    CsFensUusk(:,:,t) = sum(conj(sens(:,:,tmp_sli,:)) .* FensUusk(:,:,t,:),4);
end

%%% Fxyin* operator
FxyinsCsFensUssk=fftshift(fftshift(fft2(ifftshift(ifftshift(CsFensUusk,1),2)),1),2);

%%% Mxy* operator
kr_vec = linspace(-pi,pi-2*pi*(1/nlin),nlin); kr_mtx = repmat(kr_vec.',1,ncol);
kc_vec = linspace(-pi,pi-2*pi*(1/ncol),ncol); kc_mtx = repmat(kc_vec,nlin,1);
kspace_2d = cat(3,kr_mtx, kc_mtx);
MxysFxyinsCsFensUssk = zeros(nlin,ncol,tls);
for t = 1:tls
    dx = theta(t,1);
    dy = theta(t,2);
    mmtx_sli = cat(3, repmat(dx,nlin,ncol), repmat(dy,nlin,ncol));
    Mxy_sli = exp(1i * sum(kspace_2d.*mmtx_sli,3) );
    MxysFxyinsCsFensUssk(:,:,t) = Mxy_sli .* FxyinsCsFensUssk(:,:,t);
end

%%% Fxy* operator
FxysMxysFxyinsCsFensUssk = fftshift(fftshift(  ifft2(...
    ifftshift(ifftshift(  MxysFxyinsCsFensUssk, 1), 2)), 1), 2);

% RMk stands for "reverse model k", which will be a sum of the effects of
% each kspace shot on the image
RMk_all = zeros(numel(sli_pxls_in)*nsli,1,tls);
parfor t = 1:tls
    
    tmp_sli = sli_traj(t) + pad;
    
    % create new nx_pxls and sli_pxls
    yaw    = yaw_v(t);
    pitch  = pitch_v(t);
    roll   = roll_v(t);
    
    temp_vol2 = zeros(nlin,ncol,nsli);
    temp_vol2(recon_vxls) = 1;
    temp_vol2 = rot3d(temp_vol2,yaw,pitch,roll);
    nz_pxls_pre_rot = find(temp_vol2);
    nz_pxls_all = union(nz_pxls_pre_rot,recon_vxls); % not sure if we need nz_pxls_in here
    
    nz_im = zeros(nlin,ncol,nsli); nz_im(nz_pxls_all) = 1;
    sli_pxls_all = find(sum(nz_im,3));
    
    
    %%% Uss* operator
    tmp_imsp = zeros(nlin,ncol,nsli);
    tmp_imsp(:,:,tmp_sli) = FxysMxysFxyinsCsFensUssk(:,:,t);
    
    %%% Fz-* operator
    tmp_imsp2 = reshape(tmp_imsp,nlin*ncol,nsli);
    tmp_imsp2 = tmp_imsp2(sli_pxls_all,:);
    obj_xy_kz = fftshift(fft(ifftshift(tmp_imsp2,2), nsli, 2),2);
    
    %%% Mz* operator
    dz = dz_v(t);
    kp_vec = linspace(-pi,pi-2*pi*(1/nsli),nsli);
    p_ph = exp(1i * kp_vec     * dz).';        % par phase
    Mz_obj_xy_kz = permute(repmat(p_ph,1,numel(sli_pxls_all)),[ 2 1]) .* obj_xy_kz;
    
    %%% Fz* operator
    obj_xyz = fftshift(ifft(ifftshift(Mz_obj_xy_kz,2), nsli, 2) ,2);
    
    %%% R* operator
    temp_vol = zeros(nlin*ncol,nsli);
    temp_vol(sli_pxls_all, :) = obj_xyz;
    temp_vol = reshape(temp_vol, nlin, ncol, nsli);
    
    inv_rot_vol = rot3d(temp_vol,-1*yaw,-1*pitch,-1*roll);
    
    RMk_all(:,t) = reshape(inv_rot_vol(recon_vxls),numel(recon_vxls),1);
end
RMk = sum(RMk_all,3);
imdata = RMk;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%    E*E, used when solving normal equations                  %%%%%%%%%%
function [output] = EHE(x,U,sens,tse_traj,theta,lambda,recon_vxls )

Ex = E(x,U,sens,tse_traj,theta,recon_vxls);
EHEx = EH(Ex,U,sens,tse_traj,theta,recon_vxls);
output = EHEx + lambda;

end

function [rvol] = rot3d(vol,yaw,pitch,roll)

temp_vol = vol;
[nlin, ncol, ~] = size(vol);
if roll

    pad_r = round(nlin/2);
    pad_c = round(ncol/2);
    padsize = [pad_r, pad_c,0];
    temp_vol_big = padarray(temp_vol,padsize);
    temp_vol_big_shift = circshift(temp_vol_big,[0,0]);
    temp_vol_big_shift_rot = imrotate(temp_vol_big_shift,roll,'bilinear','crop');
    temp_vol_big_shift_rot_shift = circshift(temp_vol_big_shift_rot,[0,0]);
    temp_vol = temp_vol_big_shift_rot_shift(pad_r+1:end-pad_r,...
        pad_c+1:end-pad_c,:);  
    
end
if yaw
        temp_vol = permute( ...
            imrotate(permute(temp_vol, [3 2 1]),yaw,'bilinear','crop'), [3 2 1]);
end
if pitch
    temp_vol = permute( ...    
        imrotate(permute(temp_vol, [3 1 2]),pitch,'bilinear','crop'), [2 3 1]);
end

rvol = temp_vol;
end








