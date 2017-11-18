function [ fit] = tamer_fm_shot( dTheta_in, mt2corr, theta_prev, sens, kdata, ...
    tse_traj, U ,  msk_vxls, xprev, kfilt,rI_off,kTheta_in,cvar)
%
% TAMER forward model for a given shot, a given x and theta
%
% INPUTS:
%
%   dTheta_in: delta theta (change in motion trajectory vector)
%   mt2corr: specifies which motion parameters to correct
%   th_prev: previous estimate of theta
%   sens: sensitivity maps
%   kdata: k-space data
%   tse_traj: TSE shot trajectory (sampling trajectory)
%   U: undersampling matrix
%   msk_vxls: voxels of mask
%   xprev: previous estimate of the image x
%   kfilt: k-space weighting filter on the L2 norm
%   rI_off: offset to the axis of rotation about the I-S direction
%   kTheta_in: k-space data for motion theta at all shots other than the
%       one being evaluated
%   cvar: current varible in the gradient being calculated, i.e. which shot
%       is being updated
%
% OUTPUTS:
%
%   fit: data consistency fit

%%                             Precomputations                           %%


% reshape motion vectors
dM_in_all = zeros(numel(theta_prev),1);
dM_in_all(mt2corr) = dTheta_in;
dM_in_all_mtx = reshape(dM_in_all, size(theta_prev));
Ms = theta_prev + dM_in_all_mtx;

kdata = kdata .*U;

cor_shots = mod(mt2corr(cvar)-1,size(theta_prev,1))+1; 

%% project back
ks = E(xprev(msk_vxls),U,sens,tse_traj,Ms,msk_vxls,rI_off,cor_shots,kTheta_in);

% weight the kspace data
km_hf = kdata .* kfilt;
ks_hf = ks .* kfilt;

fit = norm(ks_hf(:)-km_hf(:))/norm(km_hf(:));

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%              E                       %%%%%%%%%%%%%%%%%%%%%%%%%%%
function [kdata] = E(x,U,sens,tse_traj,theta,msk_vxls,rI_off,cor_shots,kTheta_in)

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


Cfull2_sli = zeros(nlin,ncol,1,ncha);
for t = cor_shots
    tmp_sli = tse_traj(t,1) + pad;
    Cfull2_sli(:,:,1,:) = squeeze(sens(:,:,tmp_sli,:));
end


for t = cor_shots
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
    temp_vol(msk_vxls) = x;
    temp_volR = rot3d(temp_vol,yaw,pitch,roll,rI_off);
    
    nz_pxls_rot = find(repmat(sum(abs(temp_volR),3), 1, 1, nsli));
    nz_pxls_shot = union(msk_vxls,nz_pxls_rot);
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
    Cx = squeeze(Cfull2_sli(:,:,1,:)) .* repmat(FxyinMxyFxyUssFzinMzFzRx,1,1,ncha);
    
    % Fen
    FenCx = fftshift(fftshift(fft2(...
        ifftshift(ifftshift(Cx,1),2) ) ,1),2);
    
    % Uss
    FMx_input = FenCx(tmp_tse_traj,:,:);
    FMx_input_mtx(:,:,:,t) = FMx_input;
      
end

for t = cor_shots
    tmp_sli = tse_traj(t,1) + pad;
    tmp_tse_traj = tse_traj_mtx(t,:);
    FMx(tmp_tse_traj,:,tmp_sli,:) = FMx_input_mtx(:,:,:,t);
end

kdata = kTheta_in;
kdata(find(FMx)) = FMx(find(FMx));

end


function [rvol] = rot3d(vol,yaw,pitch,roll, rI_off)

temp_vol = vol;
[nlin, ncol, ~] = size(vol);
if roll

    pad_r = round(nlin/2);
    pad_c = round(ncol/2);
    padsize = [pad_r, pad_c,0];
    temp_vol_big = padarray(temp_vol,padsize);
    temp_vol_big_shift = circshift(temp_vol_big,[rI_off,0]);
    temp_vol_big_shift_rot = imrotate(temp_vol_big_shift,roll,'bilinear','crop');
    temp_vol_big_shift_rot_shift = circshift(temp_vol_big_shift_rot,[-rI_off,0]);
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













