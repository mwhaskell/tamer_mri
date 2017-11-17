function [ fit, grad] = mt_fit_fcn_v9p_grad_tarSh( dM_in, dM_in_indices,...
    Mn, Cfull, km, tse_traj, U , tar_pxls , full_msk_pxls, kfilter,...
    exp_str,exp_path, save_updates, rI_off)

global tamer_vars

if tamer_vars.track_opt == true;
    tamer_vars.nmotion_traj_attmpt = tamer_vars.nmotion_traj_attmpt + 1;
end

cup = tamer_vars.cup;
call = tamer_vars.call;
xprev_best = tamer_vars.xprev_best;
fit_init = tamer_vars.fit_init;

citer_off = tamer_vars.citer_vals_all(call+1,:);
msk = zeros(size(Cfull, 1), size(Cfull, 2), size(Cfull, 3));
msk(tar_pxls) = 1;

if strcmp(tamer_vars.pxl_sel_method,'cmRot')
    citer_off = [citer_off(2),citer_off(1)]; % tmp 3/28, could do this properly
                                             % but for now just rotating CM
                                             % shifts
end

if ~isequal(tar_pxls, full_msk_pxls) 
%     msk = imtranslate(msk, citer_off);

    % change made 3/27 so that the mask wraps when it shifts, but switching
    % the indices to keep the same directionality used with imtranslate,
    % though this could be changed in later versions (would also need to
    % change how tamer_vars.citer_vals_all is constructed)
    msk = circshift(msk, [citer_off(2),citer_off(1)]);
end
tar_pxls = find(msk(:));

tamer_vars.tar_pxl_all = union(tamer_vars.tar_pxl_all , tar_pxls);
per_tar = numel(tamer_vars.tar_pxl_all)/numel(full_msk_pxls);
tamer_vars.per_tar_v = [tamer_vars.per_tar_v, per_tar];

% copy best so far
xprev = xprev_best;
[ fit, xprev_new] = mt_fit_fcn_v9p( dM_in, dM_in_indices, Mn, Cfull, km, ...
    tse_traj, U , tar_pxls , full_msk_pxls, xprev, [], kfilter, rI_off);

% update best so far x and incr update variable
if (fit < fit_init)
    
    tamer_vars.fit_init = fit;
    tamer_vars.xprev_best = xprev_new;
    tamer_vars.cup = cup + 1;
    
    if save_updates
        save(strcat(exp_path,exp_str,sprintf('_update_%d.mat', cup)),...
            'xprev_new', 'fit_init', 'dM_in','tar_pxls')
    end
end

%% calculate the gradient if specified using the "fixed" version of the
%  objective function where the image is fixed and only the motion changes
%  to speed up the gradient calculation
if nargout > 1
    if tamer_vars.track_opt == true;
        tamer_vars.ngrad_calc = tamer_vars.ngrad_calc + 1;
    end
    
    [ fitng, ks] = mt_fit_fcn_v9_fixed( dM_in, dM_in_indices, Mn, Cfull, km, ...
        tse_traj, U , full_msk_pxls, xprev, [], kfilter, rI_off);
    
    grad = zeros(length(dM_in),1);
    delta = 1e-6; % changed to 1e-3 from 1e-6 on 10/27/16, changed back to 1e-6 10/28
    
    parfor cvar = 1:length(dM_in)
        
        dM_dif = dM_in;
        dM_dif(cvar) = dM_dif(cvar) + delta;
        
        [ fitg] = mt_fit_fcn_v9_fixed_shot( dM_dif, dM_in_indices, Mn, Cfull, km, ...
            tse_traj, U , full_msk_pxls, xprev, [], kfilter,rI_off,ks,cvar);
        
        grad(cvar) = (fitg - fitng) / delta;
        
    end
    
else
    grad = [];
end

tamer_vars.call = mod(call + 1, size(tamer_vars.citer_vals_all,1));

