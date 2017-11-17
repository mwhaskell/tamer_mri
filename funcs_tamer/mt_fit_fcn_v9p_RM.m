function [ RM1, RM2] = mt_fit_fcn_v9p_RM( dM_in, dM_in_indices, Mn, Cfull, km, ...
    tse_traj, U , tar_pxls , full_msk_pxls, xprev, exp_str, kfilter, rI_off, Uset)


%%                             Precomputations                           %%

%%% Currently hardcoded values
iters = 20;
lambda = 0;


[nlin, ncol, nsli, ~] = size(U);
fixed_pxls = setdiff(full_msk_pxls,tar_pxls);

% reshape motion vectors
dM_in_all = zeros(numel(Mn),1);
dM_in_all(dM_in_indices) = dM_in;
dM_in_all_mtx = reshape(dM_in_all, size(Mn));
Ms = Mn + dM_in_all_mtx;

% find R
if numel(U) == numel(find(U)), R = 1;
else, R = round(numel(U)/numel(find(U))); end
km = km .*U;

% find sequence parameters
TF = size(tse_traj,2) - 1;
tls = size(tse_traj,1);
sps = nlin/(R*TF);
pad = ( nsli - tls/sps )/2;

%% pcg
% xs_v_f = xprev(fixed_pxls);
% 
% Afxf = A(xs_v_f,U,Cfull,tse_traj,Ms,fixed_pxls,rI_off,Uset{1}) + ...
%     A(xs_v_f,U,Cfull,tse_traj,zeros(size(Ms)),fixed_pxls,rI_off,Uset{2});
% 
% 
% AtsAfxf = Astar(Afxf,U,Cfull,tse_traj,Ms,tar_pxls,rI_off,Uset{1}) + ...
%     Astar(Afxf,U,Cfull,tse_traj,zeros(size(Ms)),tar_pxls,rI_off,Uset{2});

% RHS = Astar(km,U,Cfull,tse_traj,Ms,tar_pxls,rI_off,Uset{1}) + ...
%     Astar(km,U,Cfull,tse_traj,zeros(size(Ms)),tar_pxls,rI_off,Uset{2}) - AtsAfxf;

RM1 = Astar(km,U,Cfull,tse_traj,Ms,tar_pxls,rI_off,Uset{1}) ;
RM2 = Astar(km,U,Cfull,tse_traj,zeros(size(Ms)),tar_pxls,rI_off,Uset{2});



% 
% if (~isempty(xprev))
% [xs_v_t, f, rr, it] = pcg(@(x)...
%     LHS(x,U,Cfull,tse_traj,Ms,lambda,tar_pxls,rI_off,Uset), RHS, 1e-3, iters, [], [],...
%         reshape(xprev(tar_pxls),numel(tar_pxls),1));
% else
% [xs_v_t, f, rr, it] = pcg(@(x)...
%     LHS(x,U,Cfull,tse_traj,Ms,lambda,tar_pxls,rI_off,Uset), RHS, 1e-3, iters);
% end
% pcg_out = [f, rr, it];
% 
% %%                   Evaluate Forward Model                              %%
% 
% xs_v_vol = zeros(nlin,ncol,nsli);
% xs_v_vol(fixed_pxls) = xs_v_f; xs_v_vol(tar_pxls) = xs_v_t;
% xs_v_all = xs_v_vol(full_msk_pxls);
% 
% pxl_per_sli = numel(full_msk_pxls)/nsli;
% xs_v = zeros(numel(full_msk_pxls),1);
% xs_v(pad*pxl_per_sli+1:end-pad*pxl_per_sli) = xs_v_all(pad*pxl_per_sli+1:end-pad*pxl_per_sli);
% 
% %% view reconstructed image for debugging
% x = zeros(nlin,ncol,nsli);
% x(full_msk_pxls) = xs_v;
% 
% %% project back
% ks = A(xs_v,U,Cfull,tse_traj,Ms,full_msk_pxls,rI_off,Uset{1}) + ...
%     A(xs_v,U,Cfull,tse_traj,zeros(size(Ms)),full_msk_pxls,rI_off,Uset{2});
% 
% % weight the kspace data
% km_hf = km .* kfilter;
% ks_hf = ks .* kfilter;
% 
% fit_hf = norm(ks_hf(:)-km_hf(:))/norm(km_hf(:));
% 
% if (~isempty(exp_str))
%     save(strcat(exp_str,'_tmp.mat'),'Mn','dM_in','fit_hr')
% end
% 
% 
% %% update tamer_vars
% global tamer_vars
% if tamer_vars.track_opt == true
%     tamer_vars.nobjfnc_calls = tamer_vars.nobjfnc_calls + 1;
%     tamer_vars.pcg_steps = [tamer_vars.pcg_steps, it];
%     tamer_vars.ntotal_pcg_steps = tamer_vars.ntotal_pcg_steps + it;
%     tamer_vars.fit_vec = [tamer_vars.fit_vec, fit_hf];
% end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%              A                       %%%%%%%%%%%%%%%%%%%%%%%%%%%
function [kdata] = A(x,U,Cfull,tse_traj,Ms,nz_pxls_in,rI_off, Uover)

%% precomputations
[nlin, ncol, nsli, ncha] = size(U);

% find R
if numel(U) == numel(find(U)), R = 1;
else, R = round(numel(U)/numel(find(U))); end

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

kdata = FMx .* Uover;

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%              Astar                   %%%%%%%%%%%%%%%%%%%%%%%%%%%
function [imdata] = Astar(k,U,Cfull,tse_traj,Ms,nz_pxls_in, rI_off,Uover)

k = k .* Uover;

%% precomputations
[nlin, ncol, nsli, ncha] = size(U);

nz_im = zeros(nlin,ncol,nsli); nz_im(nz_pxls_in) = 1;
sli_pxls_in = find(nz_im(:,:,round(end/2)));

% find R
if numel(U) == numel(find(U)), R = 1;
else, R = round(numel(U)/numel(find(U))); end

% find sequence parameters
TF = size(tse_traj,2) - 1;
tls = size(tse_traj,1);
sps = nlin/(R*TF);
pad = ( nsli - tls/sps )/2;


%% begin reverse model

% prep variables for parellelization
sli_traj = tse_traj(:,1); shot_traj = tse_traj(:,2:end);
dz_v = Ms(:,3); yaw_v = Ms(:,4); pitch_v = Ms(:,5); roll_v = Ms(:,6);

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
    CsFensUusk(:,:,t) = sum(conj(Cfull(:,:,tmp_sli,:)) .* FensUusk(:,:,t,:),4);
end

%%% Fxyin* operator
FxyinsCsFensUssk=fftshift(fftshift(fft2(ifftshift(ifftshift(CsFensUusk,1),2)),1),2);

%%% Mxy* operator
kr_vec = linspace(-pi,pi-2*pi*(1/nlin),nlin); kr_mtx = repmat(kr_vec.',1,ncol);
kc_vec = linspace(-pi,pi-2*pi*(1/ncol),ncol); kc_mtx = repmat(kc_vec,nlin,1);
kspace_2d = cat(3,kr_mtx, kc_mtx);
MxysFxyinsCsFensUssk = zeros(nlin,ncol,tls);
for t = 1:tls
    dx = Ms(t,1);
    dy = Ms(t,2);
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
    temp_vol2(nz_pxls_in) = 1;
    temp_vol2 = MHrot3d(temp_vol2,yaw,pitch,roll, rI_off);
    nz_pxls_pre_rot = find(temp_vol2);
    nz_pxls_all = union(nz_pxls_pre_rot,nz_pxls_in); % not sure if we need nz_pxls_in here
    
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
    
    inv_rot_vol = MHrot3d(temp_vol,-1*yaw,-1*pitch,-1*roll, rI_off);
    
    RMk_all(:,t) = reshape(inv_rot_vol(nz_pxls_in),numel(nz_pxls_in),1);
end
RMk = sum(RMk_all,3);
imdata = RMk;

end


%%           LHS function                                        %%%%%%%%%%
function [output] = LHS(x,U,Cfull,tse_traj,Ms,lambda,nz_pxls,rI_off,Uset)

Ax = A(x,U,Cfull,tse_traj,Ms,nz_pxls,rI_off,Uset{1}) + ...
    A(x,U,Cfull,tse_traj,zeros(size(Ms)),nz_pxls,rI_off,Uset{2});

AsAx = Astar(Ax,U,Cfull,tse_traj,Ms,nz_pxls,rI_off,Uset{1}) + ...
    Astar(Ax,U,Cfull,tse_traj,zeros(size(Ms)),nz_pxls,rI_off,Uset{2});

output = AsAx + lambda;

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










