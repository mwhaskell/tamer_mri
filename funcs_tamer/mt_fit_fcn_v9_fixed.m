function [ fit_hf, ks] = mt_fit_fcn_v9_fixed( dM_in, dM_in_indices, Mn, Cfull, km, ...
    tse_traj, U ,  full_msk_pxls, xprev, exp_str, kfilter, rI_off)

%%                             Precomputations                           %%



% reshape motion vectors
dM_in_all = zeros(numel(Mn),1);
dM_in_all(dM_in_indices) = dM_in;
dM_in_all_mtx = reshape(dM_in_all, size(Mn));
Ms = Mn + dM_in_all_mtx;

km = km .*U;

%% project back
ks = A(xprev(full_msk_pxls),U,Cfull,tse_traj,Ms,full_msk_pxls, rI_off);

% weight the kspace data
km_hf = km .* kfilter;
ks_hf = ks .* kfilter;

fit_hf = norm(ks_hf(:)-km_hf(:))/norm(km_hf(:));


if (~isempty(exp_str))
    save(strcat(exp_str,'_fixed_hr_tmp.mat'),'Mn','dM_in','fit')
end

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%              A                       %%%%%%%%%%%%%%%%%%%%%%%%%%%
function [kdata] = A(x,U,Cfull,tse_traj,Ms,nz_pxls_in, rI_off)

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

dx_v = Ms(:,1);dy_v = Ms(:,2);dz_v = Ms(:,3);
yaw_v = Ms(:,4); pitch_v = Ms(:,5); roll_v = Ms(:,6);
tse_traj_mtx = tse_traj(:,2:end);


Cfull2 = zeros(nlin,ncol,tls,ncha);
for t = 1:tls
    tmp_sli = tse_traj(t,1) + pad;
    Cfull2(:,:,t,:) = Cfull(:,:,tmp_sli,:);
end


for t = 1:tls
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
    temp_vol(nz_pxls_in) = x;
    temp_volR = MHrot3d(temp_vol,yaw,pitch,roll, rI_off);
    
    nz_pxls_rot = find(repmat(sum(abs(temp_volR),3), 1, 1, nsli));
    nz_pxls_shot = union(nz_pxls_in,nz_pxls_rot);
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



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%   Steve rotate function   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [rvol] = MHrot3d(vol,yaw,pitch,roll, rI_off)

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













