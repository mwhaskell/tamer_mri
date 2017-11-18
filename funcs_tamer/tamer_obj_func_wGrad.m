function [ fit, grad] = tamer_obj_func_wGrad( dTheta_in, mt2corr,...
    theta_prev, sens, kdata, tse_traj, U , tar_vxls , msk_vxls, kfilt, rI_off)
%
% TAMER objective function and gradient calculation
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
%   tar_vxls: target voxels to reconstruct
%   msk_vxls: voxels of mask
%   kfilt: k-space weighting filter on the L2 norm
%   rI_off: offset to the axis of rotation about the I-S direction
%
% OUTPUTS:
%
%   fit: data consistency fit
%   grad: motion search gradient

global tamer_vars

cup = tamer_vars.cup;
call = tamer_vars.call;
xprev_best = tamer_vars.xprev_best;
fit_init = tamer_vars.fit_init;

citer_off = tamer_vars.citer_vals_all(call+1,:);
msk = zeros(size(sens, 1), size(sens, 2), size(sens, 3));
msk(tar_vxls) = 1;

if ~isequal(tar_vxls, msk_vxls) 
    msk = circshift(msk, [citer_off(2),citer_off(1)]);
end
tar_vxls = find(msk(:));

% copy best so far
xprev = xprev_best;
[ fit, xprev_new] = tamer_obj_func_par( dTheta_in, mt2corr, theta_prev, sens, kdata, ...
    tse_traj, U , tar_vxls , msk_vxls, xprev, kfilt, rI_off);

% update best so far x and incr update variable
if (fit < fit_init)
    
    tamer_vars.fit_init = fit;
    tamer_vars.xprev_best = xprev_new;
    tamer_vars.cup = cup + 1;
    
end

%% calculate the gradient if specified using the "fixed" version of the
%  objective function where the image is fixed and only the motion changes
%  to speed up the gradient calculation
if nargout > 1
    
    [ fitng, ktheta] = tamer_fm_par( dTheta_in, mt2corr, theta_prev, sens, kdata, ...
        tse_traj, U , msk_vxls, xprev, kfilt, rI_off);
    
    grad = zeros(length(dTheta_in),1);
    delta = 1e-6; 
    
    parfor cvar = 1:length(dTheta_in)
        
        dM_dif = dTheta_in;
        dM_dif(cvar) = dM_dif(cvar) + delta;
        
        [ fitg] = tamer_fm_shot( dM_dif, mt2corr, theta_prev, sens, kdata, ...
            tse_traj, U , msk_vxls, xprev, kfilt,rI_off,ktheta,cvar);
        
        grad(cvar) = (fitg - fitng) / delta;
        
    end
    
else
    grad = [];
end

tamer_vars.call = mod(call + 1, size(tamer_vars.citer_vals_all,1));

