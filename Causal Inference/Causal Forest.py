import os
import pandas as pd
import numpy as np
from itertools import combinations
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")

data_path = r'C:\Users\16432\Desktop\Year_2022.csv'
result_path = r'C:\Users\16432\Desktop\causal_forest_ATE_results_with_pval_2022.csv'
individual_effect_dir = r'C:\Users\16432\Desktop\individual_effects_2022'
os.makedirs(individual_effect_dir, exist_ok=True)

df = pd.read_csv(data_path).dropna()

pollutants = ['Day_CO', 'Day_NO2', 'Day_O3', 'Day_PM10', 'Day_PM2.5', 'Day_SO2']

cols_to_exclude = ['Death', "climate indicator"] + pollutants 
W_raw = df.drop(columns=cols_to_exclude, errors='ignore').copy()

cat_cols = W_raw.select_dtypes(include=['object', 'category']).columns.tolist()
W_df = pd.get_dummies(W_raw, columns=cat_cols, drop_first=True)

W = W_df.values                        
y = df['Death'].values.ravel()            

if os.path.exists(result_path):
    res_df = pd.read_csv(result_path, encoding='utf-8-sig')
    done_combinations = set(res_df['combination'].tolist())
    results = res_df.to_dict('records')
else:
    done_combinations = set()
    results = []

all_combs = [comb for k in range(1, len(pollutants) + 1) for comb in combinations(pollutants, k)]
total = len(all_combs)

for idx, comb in enumerate(all_combs, start=1):
    comb_str = ','.join(comb)
    comb_col = '_'.join(comb)
    individual_file = os.path.join(individual_effect_dir, f"individual_effects_{comb_col}.csv")

    if comb_str in done_combinations and os.path.exists(individual_file):
        continue

    try:
        T = df[list(comb)].to_numpy(copy=True).astype(float)  

        model_y = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=420, n_jobs=-1)
        model_t = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=420, n_jobs=-1)

        cf = CausalForestDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=False,
            n_estimators=1000,
            random_state=420
        )
        cf.fit(Y=y, T=T, X=W)

        d = len(comb)
        ate_val = np.full(d, np.nan)
        ci_halfwidth = np.full(d, np.nan)
        p_values = np.full(d, np.nan)

        if d > 1:
            me = cf.const_marginal_effect(X=W)  
            me = np.array(me)
            if me.ndim == 1:
                me = me.reshape(-1, 1)
            m = me.shape[1]
            if m >= d:
                ate_val[:] = np.mean(me[:, :d], axis=0)
            else:
                ate_val[:m] = np.mean(me, axis=0)

            ci_lb, ci_ub = cf.ate_interval(X=W, alpha=0.05)
            ci_lb = np.atleast_1d(ci_lb)
            ci_ub = np.atleast_1d(ci_ub)
            if ci_lb.shape[0] >= d and ci_ub.shape[0] >= d:
                ci_halfwidth[:] = (ci_ub[:d] - ci_lb[:d]) / 2
            else:
                lb0 = ci_lb.flatten()[0]
                ub0 = ci_ub.flatten()[0]
                half0 = (ub0 - lb0) / 2
                ci_halfwidth[:] = half0

            try:
                p_raw = cf.ate_inference(X=W).pvalue()
                p_raw = np.atleast_1d(p_raw)
                if p_raw.shape[0] >= d:
                    p_values[:] = p_raw[:d]
                else:
                    p_values[:] = p_raw.flatten()[0]
            except:
                p_values[:] = np.nan

        else:
            single_ate = cf.ate(X=W)
            single_ate = np.atleast_1d(single_ate).flatten()[0]
            ate_val[0] = single_ate

            ci_lb, ci_ub = cf.ate_interval(X=W, alpha=0.05)
            ci_lb = np.atleast_1d(ci_lb).flatten()[0]
            ci_ub = np.atleast_1d(ci_ub).flatten()[0]
            ci_halfwidth[0] = (ci_ub - ci_lb) / 2

            try:
                p0 = cf.ate_inference(X=W).pvalue()
                p0 = np.atleast_1d(p0).flatten()[0]
                p_values[0] = p0
            except:
                p_values[0] = np.nan

        row = {'combination': comb_str}
        total_ate = np.nanmean(ate_val)
        total_ci = np.nanmean(ci_halfwidth)
        total_p = np.nanmean([p if not pd.isna(p) else 1 for p in p_values])


        if d > 1:
            formatted_items = []
            for i, poll in enumerate(comb):
                formatted_items.append(f"{ate_val[i]:.8f}+-{ci_halfwidth[i]:.8f}")
            joined = ' '.join(formatted_items)

        row[f'ATE_{comb_col}'] = f"{total_ate:.3f} ± {total_ci:.3f}"
        row[f'p_{comb_col}'] = total_p
        row[f'CI_LB_{comb_col}'] = total_ate - total_ci
        row[f'CI_UB_{comb_col}'] = total_ate + total_ci

        for i, poll in enumerate(comb):
            a = ate_val[i]
            ci = ci_halfwidth[i]
            p = p_values[i]
            row[f'ATE_{poll}'] = f"{a:.3f} ± {ci:.3f}"
            row[f'p_{poll}'] = p
            row[f'CI_LB_{poll}'] = a - ci
            row[f'CI_UB_{poll}'] = a + ci

            if d == 1:
                print(f"    {poll}: ATE = {a:.3f} ± {ci:.3f}, p = {p if not pd.isna(p) else 'NA'}")

        results.append(row)
        pd.DataFrame(results).to_csv(result_path, index=False, encoding='utf-8-sig')

        me = cf.const_marginal_effect(X=W)
        me = np.array(me)
        if me.ndim == 1:
            me = me.reshape(-1, 1)
        n_samples, m = me.shape
        if m < d:
            temp = np.full((n_samples, d), np.nan)
            temp[:, :m] = me
            me = temp

        effect_df = pd.DataFrame({
            'SampleID': np.arange(n_samples),
            'County': df['County_Level'].values,
            'Year': df['Year'].values
        })

        overall_effect = me.mean(axis=1)
        effect_df[f'Effect_{comb_col}'] = overall_effect

        for i, poll in enumerate(comb):
            effect_df[f'Effect_{poll}'] = me[:, i]

        effect_df.to_csv(individual_file, index=False, encoding='utf-8-sig')

    except Exception as e:
        print(f'{comb_str}：{e}')
        with open("error_combinations.log", "a", encoding="utf-8") as f:

            f.write(f"{comb_str} ：{str(e)}\n")



