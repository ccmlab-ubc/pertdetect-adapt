import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import scipy.optimize
from scipy.optimize import minimize
from scipy.stats import norm
from scipy import stats
from scipy.special import logsumexp
from pybads import BADS


import os

# ================ Model Fitting ================ #

def fit_bads(model, subject_id, df, bounds, plaus_bounds, n_inits, n_sims, fit_task="both", options=None):
    subj_df = (df[df['SN'] == subject_id]).reset_index(drop=True)
    
    if fit_task == "sdt":
        subj_df = subj_df[subj_df["session"] == "sdt"]
        num_fitted = np.sum(~subj_df["no_fit"])
    elif fit_task == "adapt":
        subj_df = subj_df[subj_df["session"] == "adapt"]
        num_fitted = np.sum(~subj_df["no_fit"])
    elif fit_task == "both":
        subj_df = subj_df
        # Total number of valid trials used during optimization
        n_adapt_fitted = np.sum((subj_df["session"] == "adapt") & (~subj_df["no_fit"]))
        n_sdt_fitted = np.sum((subj_df["session"] == "sdt") & (~subj_df["no_fit"]))
        num_fitted = n_adapt_fitted + n_sdt_fitted
    
    # This allows for args other than params (x) to be passed through to bads
    target_vco = lambda x: nll_vco(x, subj_df, n_sims, fit_task)
    target_co = lambda x: nll_co(x, subj_df, n_sims, fit_task)
    target_piece = lambda x: nll_piece(x, subj_df, n_sims, fit_task)
    
    if model == "vco":
        target = target_vco
    elif model == "co":
        target = target_co
    elif model == "piece":
        target = target_piece
    elif model == "co_planning":
        target = target_co_planning

    num_params = len(bounds)
    lb = np.array([bound[0] for bound in bounds])
    ub = np.array([bound[1] for bound in bounds])
    plb = np.array([bound[0] for bound in plaus_bounds])
    pub = np.array([bound[1] for bound in plaus_bounds])

    optimize_results = []
    x_vec = np.zeros((n_inits, lb.shape[0]))
    fval_vec = np.zeros(n_inits)
    
    for opt_count in range(n_inits):
        print(f"Subject Number: {subject_id}")
        print('Running optimization ' + str(opt_count) + '...')
        # Run BADS
        x0 = np.array([np.random.uniform(low, high) for low, high in bounds])
        bads = BADS(target, x0, lb, ub, plb, pub, options=options)
        optimize_results.append(bads.optimize())
        x_vec[opt_count] = optimize_results[opt_count].x
        fval_vec[opt_count] = optimize_results[opt_count].fval
    
    # Pick out the best params based on cost fxn
    idx_best = np.argmin(fval_vec)
    result_best = optimize_results[idx_best]
    x_min = result_best['x']
    fval = result_best['fval']
    
    if fit_task == "both":
        negloglik = fval * (num_fitted)
    else:
        negloglik = fval
    aic = calc_aic(negloglik * -1, num_params)
    bic = calc_bic(negloglik * -1, num_params, num_fitted)
    
    return result_best, aic, bic
    

def nll_vco(params, df, n_sims, fit_task):
    if fit_task == "sdt":
        C, sigma_v, b = params
        B = 0.9999
    else:
        C, B, sigma_v, b = params
    
    loglik, _ = vco_ll(
        df=df,
        C=C,
        B=B,
        sigma_v=sigma_v,
        b=b,
        n_sims=n_sims,
        fit_task=fit_task)
    
    return -loglik  # minimize negative log-likelihood


def nll_co(params, df, n_sims, fit_task):
    if fit_task == "sdt":
        C, sigma_comb = params
        B = 0.9999
        b = 4.999
    else:
        C, B, sigma_comb, b = params
    
    loglik, _ = co_ll(
        df=df,
        C=C,
        B=B,
        sigma_comb=sigma_comb,
        b=b,
        n_sims=n_sims,
        fit_task=fit_task
    )
    
    return -loglik  # minimize negative log-likelihood


def nll_piece(params, df, n_sims, fit_task):
    alpha, beta, sigma_p, sigma_pert, b = params
    
    loglik, _ = piece_ll(
        df=df,
        alpha=alpha,
        beta=beta, 
        sigma_p=sigma_p,
        sigma_pert=sigma_pert,
        b=b,
        n_sims=n_sims,
        fit_task=fit_task
    )

    return -loglik


def vco_ll(df, C, B, sigma_v, b, n_sims=None, fit_task="adapt"):
    
    if fit_task == "sdt":
        sigma_f_sdt = df.loc[df["session"] == "sdt", "sigma_motor_sdt"].values[0]
    elif fit_task == "adapt":
        sigma_f_adapt = df.loc[df["session"] == "adapt", "sigma_motor_adapt"].values[0]
    elif fit_task == "both":
        sigma_f_sdt = df.loc[df["session"] == "sdt", "sigma_motor_sdt"].values[0]
        sigma_f_adapt = df.loc[df["session"] == "adapt", "sigma_motor_adapt"].values[0]

    # exact workspace width based on outlier cutoff
    workspace_width = 45.0  
    p_uniform = 1.0 / workspace_width  
    # lapse_rate = 0.02
    lapse_rate = 0.0

    # Check to see if SDT task - can make vectorized calculation
    if (fit_task == "sdt") | (fit_task == "both"):
        # Extract sdt rows
        df_sdt = df[df["session"] == "sdt"].reset_index(drop=True)
        pert_resp = df_sdt["pert_resp"].values
        no_fit_sdt = df_sdt["no_fit"].values
        
        # Closed-form computation of p(det)
        mu = df_sdt["x_f"].values + df_sdt["delta_xv"].values + b
        z_pos = (C - mu) / sigma_v
        z_neg = (-C - mu) / sigma_v
        p_det = (1 - norm.cdf(z_pos)) + norm.cdf(z_neg)
        p_det = np.clip(p_det, 1e-9, 1 - 1e-9)
        p_nodet = 1 - p_det
        # Computing vector of log-probs (hence, keeping pert_resp as vector) which get summed at end
        log_p_sdt = pert_resp * np.log(p_det) + (1 - pert_resp) * np.log(1 - p_det)

    # Otherwise, we are looking at adaptation trials and using MC sims
    if (fit_task == "adapt") | (fit_task == "both"):
        # Initialize
        log_p_adapt = []
        
        # Create adaptation data frame and extract vars
        df_adapt = df[df["session"] == "adapt"].reset_index(drop=True)
        rotation = df_adapt["delta_xv"].values
        x_f = df_adapt["x_f"].values
        vis_fb = df_adapt["vis_fb"].values
        session = df_adapt["session"].values
        no_fit_adapt = df_adapt["no_fit"].values
        
        num_trials = len(x_f)
        trialnum_start_adapt = 101
        trialnum_end_adapt = 1640
        
        # Loop through trials
        for i in range(num_trials):
            sigma_f = sigma_f_adapt
            
            # MC simulation of observations
            if vis_fb[i] == 0:
                err = np.zeros(n_sims)  # ensuring no detection on no vis fb trials
            else:
                x_v = np.random.normal(loc=x_f[i] + rotation[i], scale=sigma_v, size=n_sims)
                err = x_v - b
            det_idx = np.abs(err) > C
            p_det = (det_idx.sum()) / n_sims
            p_det = np.clip(p_det, 1e-9, 1 - 1e-9)
            p_nodet = 1 - p_det
            
            if (df_adapt.loc[i, "TN"] == trialnum_start_adapt):  
                log_p_adapt.append(0)  # bc there is no prediction for first adapt trial
            
            if df_adapt.loc[i, "TN"] == trialnum_end_adapt:
                log_p_adapt = log_p_adapt  # don't make mistake of fitting first SDT trial (x_f[i + 1]) if adapt cond is first
            else:
                # Calculate intended plan for all simulated sensory measurements
                motor_plan = np.full(n_sims, b) 
                motor_plan[det_idx] = motor_plan[det_idx] - B * err[det_idx]

                # Evaluate gaussian prob of the real reach given simulated response
                next_reach = x_f[i + 1]
                likelihood_per_sim = stats.norm.pdf(next_reach, loc=motor_plan, scale=sigma_f)

                # Average probs to find marginal likelihood for this trial
                trial_prob = np.mean(likelihood_per_sim)

                # Using lapse rate to cover catastrophically low probs from 'outliers'
                mixed_prob = (1.0 - lapse_rate) * trial_prob + (lapse_rate * p_uniform)
                log_p_adapt.append(np.log(max(mixed_prob, 1e-9)))
    
    if fit_task == "adapt":
        log_p_adapt = np.array(log_p_adapt)  # bc it starts as list
        log_p_adapt[no_fit_adapt] = 0
        per_trial_logp = log_p_adapt
        total_log_p = log_p_adapt.sum()
    elif fit_task == "sdt":
        log_p_sdt[no_fit_sdt] = 0
        per_trial_logp = log_p_sdt
        total_log_p = log_p_sdt.sum()
    elif fit_task == "both":
        # Don't want zeros if we're averaging log-likes
        log_p_adapt = np.array(log_p_adapt)
        adapt_ll = log_p_adapt[~no_fit_adapt]
        sdt_ll = log_p_sdt[~no_fit_sdt]
        
        # Taking the mean of log-likes so that there is no task advantage from num of trials
        mean_adapt = np.mean(adapt_ll) 
        mean_sdt = np.mean(sdt_ll)
        per_trial_logp = 0
        # Divide by 2 to get average log per trial
        total_log_p = (mean_adapt + mean_sdt) / 2.0

    return total_log_p, per_trial_logp


def co_ll(df, C, B, sigma_comb, b, n_sims, fit_task):
    
    if fit_task == "sdt":
        sigma_f_sdt = df.loc[df["session"] == "sdt", "sigma_motor_sdt"].values[0]
    elif fit_task == "adapt":
        sigma_f_adapt = df.loc[df["session"] == "adapt", "sigma_motor_adapt"].values[0]
    elif fit_task == "both":
        sigma_f_sdt = df.loc[df["session"] == "sdt", "sigma_motor_sdt"].values[0]
        sigma_f_adapt = df.loc[df["session"] == "adapt", "sigma_motor_adapt"].values[0]

    # exact workspace width based on outlier cutoff
    workspace_width = 45.0  
    p_uniform = 1.0 / workspace_width  
    # lapse_rate = 0.02
    lapse_rate = 0.0

    # Check to see if SDT task - can make vectorized calculation
    if (fit_task == "sdt") | (fit_task == "both"):
        # Extract sdt rows
        df_sdt = df[df["session"] == "sdt"].reset_index(drop=True)
        pert_resp = df_sdt["pert_resp"].values
        no_fit_sdt = df_sdt["no_fit"].values
        
        # Closed-form computation of p(det)
        mu = df_sdt["delta_xv"].values 
        z_pos = (C - mu) / sigma_comb
        z_neg = (-C - mu) / sigma_comb
        p_det = (1 - norm.cdf(z_pos)) + norm.cdf(z_neg)
        p_det = np.clip(p_det, 1e-9, 1 - 1e-9)
        p_nodet = 1 - p_det

        # Computing vector of log-probs (hence, keeping pert_resp as vector) which get summed at end
        log_p_sdt = pert_resp * np.log(p_det) + (1 - pert_resp) * np.log(1 - p_det)

    # Otherwise, we are looking at adaptation trials and using MC sims
    if (fit_task == "adapt") | (fit_task == "both"):
        # Initialize
        log_p_adapt = []
        
        # Create adaptation data frame and extract vars
        df_adapt = df[df["session"] == "adapt"].reset_index(drop=True)
        rotation = df_adapt["delta_xv"].values
        x_f = df_adapt["x_f"].values
        vis_fb = df_adapt["vis_fb"].values
        session = df_adapt["session"].values
        no_fit_adapt = df_adapt["no_fit"].values
        
        num_trials = len(x_f)
        trialnum_start_adapt = 101
        trialnum_end_adapt = 1640
        
        # Loop through trials
        for i in range(num_trials):
            sigma_f = sigma_f_adapt
            
            # MC simulation of observations
            if vis_fb[i] == 0:
                err = np.zeros(n_sims)
            else:
                # Bc xv - xp is sum of RVs whose means are xf + rot and xf
                err = np.random.normal(loc=rotation[i], scale=sigma_comb, size=n_sims)
            
            det_idx = np.abs(err) > C
            p_det = (det_idx.sum()) / n_sims
            p_det = np.clip(p_det, 1e-9, 1 - 1e-9)
            p_nodet = 1 - p_det
   
            if (df_adapt.loc[i, "TN"] == trialnum_start_adapt):  
                log_p_adapt.append(0)  # bc there is no prediction for first adapt trial
            
            if df_adapt.loc[i, "TN"] == trialnum_end_adapt:
                log_p_adapt = log_p_adapt  # don't make mistake of fitting first SDT trial (x_f[i + 1]) if adapt cond is first
            else:
                # Calculate intended plan for all simulated sensory measurements
                motor_plan = np.full(n_sims, b) 
                motor_plan[det_idx] = motor_plan[det_idx] - B * err[det_idx]

                # Evaluate gaussian prob of the real reach given simulated response
                next_reach = x_f[i + 1]
                likelihood_per_sim = stats.norm.pdf(next_reach, loc=motor_plan, scale=sigma_f)

                # Average probs to find marginal likelihood for this trial
                trial_prob = np.mean(likelihood_per_sim)

                # Using lapse rate to cover catastrophically low probs from 'outliers'
                mixed_prob = (1.0 - lapse_rate) * trial_prob + (lapse_rate * p_uniform)
                log_p_adapt.append(np.log(max(mixed_prob, 1e-9)))
    
    if fit_task == "adapt":
        log_p_adapt = np.array(log_p_adapt)  # bc it starts as list
        log_p_adapt[no_fit_adapt] = 0
        per_trial_logp = log_p_adapt
        total_log_p = log_p_adapt.sum()
    elif fit_task == "sdt":
        log_p_sdt[no_fit_sdt] = 0
        per_trial_logp = log_p_sdt
        total_log_p = log_p_sdt.sum()
    elif fit_task == "both":
        # Don't want zeros if we're averaging log-likes
        log_p_adapt = np.array(log_p_adapt)
        adapt_ll = log_p_adapt[~no_fit_adapt]
        sdt_ll = log_p_sdt[~no_fit_sdt]
        
        # Taking the mean of log-likes so that there is no task advantage from num of trials
        mean_adapt = np.mean(adapt_ll) 
        mean_sdt = np.mean(sdt_ll)
        per_trial_logp = None  # placeholder - doesn't get used when fitting both tasks
        # Divide by 2 to get average log per trial
        total_log_p = (mean_adapt + mean_sdt) / 2.0 

    return total_log_p, per_trial_logp


def piece_ll(df, alpha, beta, sigma_p, sigma_pert, b, n_sims, fit_task, mu_pert=0):
    if fit_task == "sdt":
        sigma_f_sdt = df.loc[df["session"] == "sdt", "sigma_motor_sdt"].values[0]
    elif fit_task == "adapt":
        sigma_f_adapt = df.loc[df["session"] == "adapt", "sigma_motor_adapt"].values[0]
    else:
        sigma_f_sdt = df.loc[df["session"] == "sdt", "sigma_motor_sdt"].values[0]
        sigma_f_adapt = df.loc[df["session"] == "adapt", "sigma_motor_adapt"].values[0]
    
    rotation = df["delta_xv"].values
    x_f = df["x_f"].values
    vis_fb = df["vis_fb"].values
    pert_resp = df["pert_resp"].values
    session = df["session"].values

    num_trials = len(x_f)
    trialnum_start_adapt = 101
    trialnum_end_adapt = 1640
    
    no_fit_idx = df["no_fit"].values
    
    # Initialize 
    log_p = [] 

    # exact workspace width based on outlier cutoff
    workspace_width = 45.0  
    p_uniform = 1.0 / workspace_width  
    # lapse_rate = 0.02
    lapse_rate = 0.0
    
    # Loop through trials
    for i in range(num_trials):
        if session[i] == "adapt":
            sigma_f = sigma_f_adapt
        elif session[i] == "sdt":
            sigma_f = sigma_f_sdt
        
        # No vis fb trials only occur during adapt task
        if (vis_fb[i] == 0) & (session[i] == "adapt"):
            post_pert = np.zeros(n_sims)
            r_hat = np.zeros(n_sims)
            sigma_v = 1e2
            
        elif vis_fb[i] == 1:
            sigma_v = alpha + beta * np.abs(x_f[i] + rotation[i])
    
            # MC sampling of observations
            x_p = np.random.normal(loc=x_f[i], scale=sigma_p, size=n_sims)
            x_v = np.random.normal(loc=x_f[i] + rotation[i], scale=sigma_v, size=n_sims)

            # Compute likelihood of perturbed world state        
            num = np.exp(-(b**2 * (sigma_p**2+sigma_pert**2+sigma_v**2)+sigma_pert**2 * x_p**2
                                 + sigma_v**2 * x_p**2 + sigma_f**2 * (x_p-x_v)**2 + sigma_p**2 * x_v**2
                                 -2 * b * (sigma_pert**2 * x_p + sigma_v**2 * x_p + sigma_p**2 * x_v))
                               / (2 * (sigma_p**2 * (sigma_pert**2+sigma_v**2)+sigma_f**2 
                                       * (sigma_p**2+sigma_pert**2+sigma_v**2)))
                        )
            denom = 2 * np.pi * sigma_f * sigma_p * sigma_v * np.sqrt( 
                (1+ sigma_pert**2 / sigma_v**2) / sigma_f**2 + (sigma_p**2 + sigma_pert**2 + sigma_v**2) 
                / (sigma_p**2 * sigma_v**2)
            )
            like_pert = num / denom

            # Compute likelihood of unperturbed
            num = np.exp(-(b**2 * (sigma_p**2 + sigma_v**2) + sigma_v**2 * x_p**2 + sigma_f**2 * (x_p-x_v)**2
                            + sigma_p**2 * x_v**2 - 2 * b * (sigma_v**2 * x_p + sigma_p**2 * x_v)) 
                         / (2 * (sigma_p**2 * sigma_v**2 + sigma_f**2 * (sigma_p**2 + sigma_v**2))))
            denom = 2 * np.pi * sigma_f * sigma_p * np.sqrt( 1 / sigma_f**2 + 1 / sigma_p**2 + 1 / sigma_v**2) * sigma_v
            like_unpert = num / denom
            
            # We can do likelihood test because of flat prior
            p_det = (like_pert > like_unpert).sum() / n_sims
            p_det = np.clip(p_det, 1e-9, 1 - 1e-9)
            p_nodet = 1 - p_det
            
            if session[i] == "sdt":
                if pert_resp[i] == 1:
                    log_p.append(np.log(p_det))
                elif pert_resp[i] == 0:
                    log_p.append(np.log(p_nodet))
            elif session[i] == "adapt":
                # Posterior on rotation - adding tiny scalar to prevent 0 division
                eps = 1e-12 
                post_pert = (like_pert) / (like_pert + like_unpert + eps)
                post_pert = np.clip(post_pert, 1e-9, 1.0 - 1e-9)
                _, r_hat, sigma_rhat = prod3gauss(mu_pert, x_v, b, sigma_pert, sigma_v, sigma_f)
                
        if (session[i] == "adapt"):
            if (df.loc[i, "TN"] == trialnum_start_adapt):  # only append if sdt task was first
                log_p.append(0)  # bc there is no prediction for first adapt trial

            # Separating conditionals this way ensures that on the first adapt trial you fill
            # in a zero for trial 1 and add the p(next trial data|current trial, params, model)
            if (df.loc[i, "TN"] == trialnum_end_adapt):
                log_p = log_p  # don't make mistake of fitting first SDT trial (x_f[i + 1])
            else:
                # # Each appended log-prob is p(next trial data|current trial, params, model)
                # Calculate the 1,000 intended plans without adding execution noise
                motor_plan = -(post_pert * r_hat) + b

                # Evaluate the exact Gaussian probability of the next reach for each plan
                next_reach = x_f[i + 1]
                likelihood_per_sim = stats.norm.pdf(next_reach, loc=motor_plan, scale=sigma_f)

                # Average them and take the log
                trial_prob = np.mean(likelihood_per_sim)
                # log_p.append(np.log(max(trial_prob, 1e-9)))

                # Uncomment if using lapse rate
                mixed_prob = (1.0 - lapse_rate) * trial_prob + (lapse_rate * p_uniform)
                log_p.append(np.log(max(mixed_prob, 1e-9)))
                
                # mu_resp = pred_reach.mean()
                # sigma_resp = pred_reach.std()
                # log_p.append(-0.5 * np.log(2 * np.pi * sigma_resp**2)- ((next_reach - (mu_resp)) ** 2 / (2 * sigma_resp**2)))
               
    log_p = np.array(log_p)
    
    # Create a boolean mask of valid trials 
    valid_mask = ~no_fit_idx
    
    if fit_task == "both":
        valid_adapt_ll = log_p[(session == "adapt") & valid_mask]
        valid_sdt_ll = log_p[(session == "sdt") & valid_mask]
        
        mean_adapt = np.mean(valid_adapt_ll) if len(valid_adapt_ll) > 0 else 0
        mean_sdt = np.mean(valid_sdt_ll) if len(valid_sdt_ll) > 0 else 0
        
        # Divide by 2 to get average log per trial
        total_log_p = (mean_adapt + mean_sdt) / 2.0  
        per_trial_logp = None  # placeholder - doesn't get used when fitting both tasks
    else:
        log_p[no_fit_idx] = 0
        total_log_p = np.sum(log_p)
        per_trial_logp = log_p
        
    return total_log_p, per_trial_logp
   

def piece_simulate_adapt(alpha, beta, sigma_p, sigma_pert, b, df, mu_pert=0):
    df_adapt = df.loc[df["session"] == "adapt", :]
    # Grab vars from dataframe 
    sigma_f = df_adapt.loc[df["session"] == "adapt", "sigma_motor_adapt"].values[0]
    rotation = df_adapt["delta_xv"].values
    x_f = df_adapt["x_f"].values
    vis_fb = df_adapt["vis_fb"].values
    num_trials = len(rotation)
    
    # Initialize 
    x_f_hat = np.zeros(num_trials)
    
    # Loop through trials
    for i in range(num_trials - 1):
        if i == 0:
            x_f_hat[i] = b + np.random.normal(loc=0, scale=sigma_f)
        
        if vis_fb[i] == 1:
            sigma_v = alpha + beta * np.abs(x_f_hat[i] + rotation[i])
        else:
            sigma_v = 1e2
        
        # MC sampling of observations
        x_p = np.random.normal(loc=x_f_hat[i], scale=sigma_p)
        x_v = np.random.normal(loc=x_f_hat[i] + rotation[i], scale=sigma_v)
 
        # Compute likelihood of perturbed world state        
        num = np.exp(-(b**2 * (sigma_p**2+sigma_pert**2+sigma_v**2)+sigma_pert**2 * x_p**2
                             +sigma_v**2 * x_p**2+sigma_f**2 * (x_p-x_v)**2+sigma_p**2 * x_v**2
                             -2 * b * (sigma_pert**2 * x_p+sigma_v**2 * x_p + sigma_p**2 * x_v))
                           / (2 * (sigma_p**2 * (sigma_pert**2+sigma_v**2)+sigma_f**2 
                                   * (sigma_p**2+sigma_pert**2+sigma_v**2)))
                    )
        denom = 2 * np.pi * sigma_f * sigma_p * sigma_v * np.sqrt( 
            (1+ sigma_pert**2 / sigma_v**2) / sigma_f**2 + (sigma_p**2 + sigma_pert**2 + sigma_v**2) 
            / (sigma_p**2 * sigma_v**2)
        )
        like_pert = num / denom

        # Compute likelihood of unperturbed
        num = np.exp(-(b**2 * (sigma_p**2 + sigma_v**2) + sigma_v**2 * x_p**2 + sigma_f**2 * (x_p-x_v)**2
                        + sigma_p**2 * x_v**2 - 2 * b * (sigma_v**2 * x_p + sigma_p**2 * x_v)) 
                     / (2 * (sigma_p**2 * sigma_v**2 + sigma_f**2 * (sigma_p**2 + sigma_v**2))))
        denom = 2 * np.pi * sigma_f * sigma_p * np.sqrt( 1 / sigma_f**2 + 1 / sigma_p**2 + 1 / sigma_v**2) * sigma_v
        like_unpert = num / denom

        # Posterior on rotation
        if vis_fb[i] == 0:
            post_pert = 0
            r_hat = 0
        else:
            post_pert = (like_pert) / (like_pert + like_unpert)
            _, r_hat, sigma_rhat = prod3gauss(mu_pert, x_v, b, sigma_pert, sigma_v, sigma_f)
        x_f_hat[i + 1] = -(post_pert * r_hat) + np.random.normal(loc=b, scale=sigma_f)
   
    return x_f_hat


def piece_simulate_sdt(alpha, beta, sigma_p, sigma_pert, b, df, mu_pert=0):
    
    df_sdt = df.loc[df["session"] == "sdt", :]
    # Grab vars from dataframe 
    sigma_f = df_sdt.loc[df["session"] == "sdt", "sigma_motor_sdt"].values[0]
    rotation = df_sdt["delta_xv"].values
    x_f = df_sdt["x_f"].values
    vis_fb = df_sdt["vis_fb"].values
    num_trials = len(rotation)
    
    # Initialize 
    resp = np.zeros(num_trials)

    # Loop through trials
    for i in range(num_trials):
        sigma_v = alpha + beta * np.abs(x_f[i] + rotation[i])
        
        # Simulate one set of observations per trial
        x_p = np.random.normal(loc=x_f[i], scale=sigma_p)
        x_v = np.random.normal(loc=x_f[i] + rotation[i], scale=sigma_v)
 
        # Compute likelihood of perturbed world state        
        num = np.exp(-(b**2 * (sigma_p**2+sigma_pert**2+sigma_v**2)+sigma_pert**2 * x_p**2
                             +sigma_v**2 * x_p**2+sigma_f**2 * (x_p-x_v)**2+sigma_p**2 * x_v**2
                             -2 * b * (sigma_pert**2 * x_p+sigma_v**2 * x_p + sigma_p**2 * x_v))
                           / (2 * (sigma_p**2 * (sigma_pert**2+sigma_v**2)+sigma_f**2 
                                   * (sigma_p**2+sigma_pert**2+sigma_v**2)))
                    )
        denom = 2 * np.pi * sigma_f * sigma_p * sigma_v * np.sqrt( 
            (1+ sigma_pert**2 / sigma_v**2) / sigma_f**2 + (sigma_p**2 + sigma_pert**2 + sigma_v**2) 
            / (sigma_p**2 * sigma_v**2)
        )
        like_pert = num / denom

        # Compute likelihood of unperturbed
        num = np.exp(-(b**2 * (sigma_p**2 + sigma_v**2) + sigma_v**2 * x_p**2 + sigma_f**2 * (x_p-x_v)**2
                        + sigma_p**2 * x_v**2 - 2 * b * (sigma_v**2 * x_p + sigma_p**2 * x_v)) 
                     / (2 * (sigma_p**2 * sigma_v**2 + sigma_f**2 * (sigma_p**2 + sigma_v**2))))
        denom = 2 * np.pi * sigma_f * sigma_p * np.sqrt( 1 / sigma_f**2 + 1 / sigma_p**2 + 1 / sigma_v**2) * sigma_v
        like_unpert = num / denom

        # Response to Yes/No re: perturbation
        if like_pert > like_unpert:
            resp[i] = True
        else: 
            resp[i] = False
   
    return resp


def co_simulate_adapt(df, C, B, sigma_comb, b, sigma_plan=None):
    
    df_adapt = df.loc[df["session"] == "adapt", :]
    # Grab vars from dataframe 
    sigma_f = df_adapt.loc[df["session"] == "adapt", "sigma_motor_adapt"].values[0]
    rotation = df_adapt["delta_xv"].values
    x_f = df_adapt["x_f"].values
    vis_fb = df_adapt["vis_fb"].values
    num_trials = len(rotation)
    
    # Initialize 
    x_f_hat = np.zeros(num_trials)
    
    # Loop through trials
    for i in range(num_trials - 1):
        if i == 0:
            x_f_hat[i] = b + np.random.normal(loc=0, scale=sigma_f)
        # Simulate one set of observations per trial
        if vis_fb[i] == 1:
            err = np.random.normal(loc=rotation[i], scale=sigma_comb)
        else:
            err = 0

        if np.abs(err) > C:
            x_f_hat[i + 1] = -B * err + b + np.random.normal(loc=0, scale=sigma_f)
        else: 
            x_f_hat[i + 1] = b + np.random.normal(loc=0, scale=sigma_f) 

    return x_f_hat


def co_simulate_sdt(df, C, sigma_comb, b, sigma_plan=None):
    
    df_sdt = df.loc[df["session"] == "sdt", :]
    # Grab vars from dataframe 
    sigma_f = df_sdt.loc[df["session"] == "sdt", "sigma_motor_sdt"].values[0]
    rotation = df_sdt["delta_xv"].values
    x_f = df_sdt["x_f"].values
    vis_fb = df_sdt["vis_fb"].values
    num_trials = len(rotation)
    
    # Initialize 
    resp = np.zeros(num_trials)
    
    # Loop through trials
    for i in range(num_trials):
        # Simulate one set of observations per trial
        if vis_fb[i] == 1:
            err = np.random.normal(loc=rotation[i], scale=sigma_comb)
        else:
            err = 0
        
        if np.abs(err) > C:
            resp[i] = True
        else: 
            resp[i] = False

    return resp


def vco_simulate_adapt(df, C, B, sigma_v, b, sigma_plan=None):
    
    df_adapt = df.loc[df["session"] == "adapt", :]
    # Grab vars from dataframe 
    sigma_f = df_adapt.loc[df["session"] == "adapt", "sigma_motor_adapt"].values[0]
    x_f = df_adapt["x_f"].values
    rotation = df_adapt["delta_xv"].values
    vis_fb = df_adapt["vis_fb"].values
    num_trials = len(rotation)
    
    # Initialize 
    x_f_hat = np.zeros(num_trials)
    
    # Loop through trials
    for i in range(num_trials - 1): 
        if i == 0:
            x_f_hat[i] = b + np.random.normal(loc=0, scale=sigma_f)
        
        if vis_fb[i] == 0:
            err = 0 # ensuring no detection on no vis fb trials
        else:
            x_v = np.random.normal(loc=x_f_hat[i] + rotation[i], scale=sigma_v)
            err = x_v - b

        if np.abs(err) > C:
            x_f_hat[i + 1] = -B * err + b + np.random.normal(loc=0, scale=sigma_f)
        else: 
            x_f_hat[i + 1] = b + np.random.normal(loc=0, scale=sigma_f) 

    return x_f_hat


def vco_simulate_sdt(df, C, sigma_v, b, sigma_plan=None):
    
    df_sdt = df.loc[df["session"] == "sdt", :]
    # Grab vars from dataframe 
    sigma_f = df_sdt.loc[df["session"] == "sdt", "sigma_motor_sdt"].values[0]
    x_f = df_sdt["x_f"].values
    rotation = df_sdt["delta_xv"].values
    vis_fb = df_sdt["vis_fb"].values
    num_trials = len(rotation)
    
    # Initialize 
    resp = np.zeros(num_trials)
    
    # Loop through trials
    for i in range(num_trials):
        # Simulate one set of observations per trial
        if vis_fb[i] == 1:
            x_v = np.random.normal(loc=x_f[i] + rotation[i], scale=sigma_v)
        else:
            x_v = b
        err = x_v - b

        if np.abs(err) > C:
            resp[i] = True
        else: 
            resp[i] = False

    return resp

    
def compute_d_prime(df: pd.DataFrame, s: int, collapsed: bool, pert_start: int, subj_col: str, rot_col: str, det_col: str):

    # NOTE: Add a 'collapsed' boolean for collapsing rotations. Will be necessary for linear fitting.

    # Select subject s and only select experimental trials
    subj = df[df[subj_col] == s].iloc[pert_start:].copy()

    # Collect rotations and responses
    if collapsed:
        subj[rot_col] = subj[rot_col].abs()

    perts = subj[rot_col].unique()
    pert_responses = subj[det_col]

    # Calculate z score for FAs
    noise_trials = len(subj[subj[rot_col] == 0])
    false_alarms = len(subj[(pert_responses == 1) & (subj[rot_col] == 0)])
    false_alarm_rate = (false_alarms + 0.5) / (noise_trials + 1)
    z_false_alarm = norm.ppf(false_alarm_rate)

    # Init df for collecting d prime vals
    df_sdt = pd.DataFrame(columns=['SN', 'delta_xv', "d prime"])

    # Loops through perts and calculate d prime
    # Concat to df
    for p in perts:
        signal_trials = len(subj[subj[rot_col] == p])
        hits = len(subj[(pert_responses == 1) & (subj[rot_col] == p)])
        hit_rate = (hits + 0.5) / (signal_trials + 1)
        z_hit = norm.ppf(hit_rate)
        d_prime = z_hit - z_false_alarm

        new_row = pd.DataFrame({'SN': [int(s)], 'delta_xv': [float(p)], 'd prime': [d_prime]})
        df_sdt = pd.concat([df_sdt, new_row], ignore_index=True)
        df_sdt = df_sdt.sort_values(by="delta_xv").reset_index(drop=True)

    return df_sdt


def calculate_adaptation(df_sim, n_trials):
    x_f = df_sim['x_f'].to_numpy()
    delta_xv = df_sim['delta_xv'].to_numpy()
    adaptation = np.zeros(n_trials)

    x_f = np.append(x_f, 0)
    delta_xv = np.append(delta_xv, 0)
    adaptation = np.append(adaptation, 0)

    pert_idx = np.arange(1, n_trials, 2, dtype=int)
    adaptation[pert_idx] = x_f[pert_idx + 1] - x_f[pert_idx - 1]

    print("Modulo check:", np.argmax(np.abs(delta_xv) > 0) % 2)

    df_postpred = pd.DataFrame({
        "x_f": x_f,
        "delta_xv": delta_xv,
        "adaptation": adaptation,
    })

    df_adapt = df_postpred.iloc[1::2].reset_index(drop=True)
    return df_adapt


def calc_aic(ll, num_params):
    aic = -2 * ll + 2 * num_params
    return aic


def calc_bic(ll, num_params, num_trials):
    bic = -2 * ll + num_params * np.log(num_trials)
    return bic


def prod_gaus(mu1, mu2, sigma1, sigma2):
    mu1 = np.array(mu1)
    mu2 = np.array(mu2)
    sigma1 = np.array(sigma1)
    sigma2 = np.array(sigma2)
    
    var1 = sigma1**2
    var2 = sigma2**2
    
    # New mean
    mu_combined = (var2 * mu1 + var1 * mu2) / (var1 + var2)

    # New var and SD
    var_combined = (var1 * var2) / (var1 + var2)
    sigma_combined = np.sqrt(var_combined)

    # Integral (scaling factor) - represents overlap between two distributions.
    exponent = -((mu1 - mu2)**2) / (2 * (var1 + var2))
    normalization = 1 / np.sqrt(2 * np.pi * (var1 + var2))
    integral = normalization * np.exp(exponent)

    # peak_height corresponds to the 'pre' variable in your original code
    peak_prob = integral / (sigma_combined * np.sqrt(2 * np.pi))

    return integral, mu_combined, sigma_combined, peak_prob


def prod3gauss(mu1, mu2, mu3, sigma1, sigma2, sigma3):
    # Product (and integral) of first two gaussians
    int1, mu12, sigma12, _ = prod_gaus(mu1, mu2, sigma1, sigma2)

    # Combine result with third gaussian
    int2, mu_combined, sigma_combined, _ = prod_gaus(mu12, mu3, sigma12, sigma3)

    # Multiple integrals
    final_integral = int1 * int2

    return final_integral, mu_combined, sigma_combined


def save_csv(data, model, file_dir="model-fits/"):
    count = sum(
        1 for f in os.listdir(file_dir)
        if f.startswith(f"fit_{model}") and f.endswith(".csv")
    )
    filename = os.path.join(file_dir, f"fit_{model}_{count + 1}.csv")
    data.to_csv(filename, index=False, encoding="ascii", errors="ignore")
    return filename


def remove_outliers(z_thresh: float, df: pd.DataFrame, reach_col: str, subj_col: str, abs_cutoff=False, switch_outlier_with_mean=False):

    """
    Remove outliers from a kinematic variable using within-subject z-scoring.

    Parameters:
    ----------
    z_thresh : float
        The z-score threshold above which data points are considered outliers.
    df : pd.DataFrame
        The input DataFrame containing the kinematic data.
    reach_col : str
        The column name of the kinematic variable to clean (e.g., "theta_maxradv").

    Returns:
    -------
    pd.DataFrame
        The DataFrame with additional columns:
        - <reach_col>_z: z-scored values within each subject (subj_col).
        - <reach_col>_outlier: boolean flag for outliers.
        - <reach_col>_mean: mean of non-outlier trials within each subject.
        - <reach_col>_clean: original values with outliers replaced by NaN.
    """

    # hard limit on reach angles above 30 degrees

    # z-score hand angle data
    df[f"{reach_col}_z"] = df.groupby(subj_col)[reach_col].transform(stats.zscore)

    # Create outlier column
    df[f"{reach_col}_outlier"] = np.abs(df[f"{reach_col}_z"]) > z_thresh

    # Calculate within-subject mean using non-outlier trials only
    df[f"{reach_col}_mean"] = df[np.abs(df[f"{reach_col}_z"]) <= z_thresh].groupby(subj_col)[f"{reach_col}"].transform(np.mean)

    # Replace outliers with within-subject mean values
    df[f"{reach_col}_mean"] = df.groupby(subj_col)[f"{reach_col}_mean"].transform(lambda x: x.fillna(np.nanmean(x)))

    ## Create final column with "cleaned" hand angles OR replace outliers with NaNs
    if switch_outlier_with_mean == True:
        df[f"{reach_col}_clean"] = np.where(np.abs(df[f"{reach_col}_z"]) > z_thresh,
                                             df[f"{reach_col}_mean"],
                                             df[f"{reach_col}"])
    else:
        df[f"{reach_col}_clean"] = np.where(np.abs(df[f"{reach_col}_z"]) > z_thresh,
                                            np.nan,
                                            df[f"{reach_col}"])

    return df


def test_remove_outliers(z_thresh, df, reach_col, subj_col, task, abs_cutoff, switch_outlier_with_mean=False):

    # hard limit on reach angles above abs_cutoff degrees
    idx_valid = np.abs(df[f"{reach_col}"]) <= abs_cutoff 
    
    # Calculate z-score hand angle data using non-outlier trials only
    df[f"{reach_col}_z"] = df.loc[idx_valid].groupby(subj_col)[reach_col].transform(stats.zscore)
    # df[f"{reach_col}_z"] = df.groupby(subj_col)[reach_col].transform(stats.zscore)

    # Create outlier column
    idx_outlier = (~idx_valid) | (np.abs(df[f"{reach_col}_z"]) > z_thresh)
    df[f"{reach_col}_outlier"] = idx_outlier
    
    # Create col to indicate which trials to exclude from MLE (trials after outlier).
    # Tasks treated differently: adaptation fitting involves predicting next trial; sdt fitting involves predictin
    # current.
    if task == "adapt":
        desired_idx = [idx_outlier[i] or (idx_outlier[i - 1] if i > 0 else False) for i in range(len(idx_outlier))]
    elif task == "sdt":
        desired_idx = idx_outlier
    df["no_fit"] = desired_idx
                                                
    # Calculate within-subject mean using non-outlier trials only
    df[f"{reach_col}_mean"] = df[~idx_outlier].groupby(subj_col)[f"{reach_col}"].transform("mean")

    # Replace outliers with within-subject mean values
    df[f"{reach_col}_mean"] = df.groupby(subj_col)[f"{reach_col}_mean"].transform(lambda x: x.fillna(np.nanmean(x)))

    # Create final column with "cleaned" hand angles OR replace outliers with NaNs
    if switch_outlier_with_mean == True:
        df[f"{reach_col}_clean"] = np.where(df[f"{reach_col}_outlier"] == True,
                                             df[f"{reach_col}_mean"],
                                             df[f"{reach_col}"])
    else:
        df[f"{reach_col}_clean"] = np.where(df[f"{reach_col}_outlier"] == True,
                                            np.nan,
                                            df[f"{reach_col}"])

    return df

