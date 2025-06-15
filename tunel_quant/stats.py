'''
All statistics functions are built to accept the output of summarize_analysis,
which has the following columns:
        ['name', 'group', 'location', 'mouse', 
        'definitely alive', 'definitely dead', 'likely alive', 'likely dead']

'''

#Import necessary libraries
import itertools
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial

'''ANOVA'''

# ──────────────────────────────────────────────────────────────────
# 1.  Core ANOVA function
# ──────────────────────────────────────────────────────────────────
def anova(df,
          *,
          post_hoc: bool = False,
          include_likely: bool = True):
    """
    One-way ANOVA comparing mice across treatment groups on the proportion
    of dead nuclei, with optional Tukey post-hoc tests.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns
        ['name', 'group', 'location', 'mouse',
         'definitely alive', 'definitely dead',
         'likely alive', 'likely dead'].
        Rows can be images, slices, or any finer unit.
    post_hoc : bool, default False
        If True, performs Tukey HSD pair-wise comparisons.
    include_likely : bool, default True
        If True, adds “likely alive/dead” counts to the definite counts.

    Returns
    -------
    results : dict
        {
          "anova" : statsmodels ANOVA table (DataFrame),
          "tukey":  statsmodels TukeyHSD result object or None,
          "agg"  :  mouse-level summary DataFrame (for inspection)
        }
    """

    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # ---- 0.  basic column check ----
    need = {'definitely alive', 'definitely dead',
            'likely alive', 'likely dead',
            'group', 'mouse'}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Data frame is missing columns: {missing}")

    # ---- 1.  choose which counts to use ----
    if include_likely:
        alive = df['definitely alive'] + df['likely alive']
        dead  = df['definitely dead']  + df['likely dead']
    else:
        alive = df['definitely alive']
        dead  = df['definitely dead']

    tmp = df.copy()
    tmp['alive_total'] = alive
    tmp['dead_total']  = dead

    # ---- 2.  collapse to one row per mouse ----
    agg = (tmp.groupby(['mouse', 'group'], as_index=False)
               .agg({'alive_total': 'sum', 'dead_total': 'sum'}))
    agg['prop_dead'] = agg['dead_total'] / (agg['alive_total'] + agg['dead_total'])

    # sanity: at least two groups & >=2 mice per group for Tukey
    if post_hoc:
        cnts = agg.groupby('group')['mouse'].nunique()
        if (cnts < 2).any():
            raise ValueError("Tukey HSD needs ≥2 mice in every group.")

    # ---- 3.  one-way ANOVA on mouse-level proportions ----
    model = ols('prop_dead ~ C(group)', data=agg).fit()
    anova_tbl = sm.stats.anova_lm(model, typ=2)   # Type-II SS

    # ---- 4.  optional Tukey post hoc ----
    tukey_res = None
    if post_hoc:
        tukey_res = pairwise_tukeyhsd(agg['prop_dead'], agg['group'])

    return {"anova": anova_tbl, "tukey": tukey_res, "agg": agg}

# ──────────────────────────────────────────────────────────────────
# 2.  Helper: Tukey → DataFrame
# ──────────────────────────────────────────────────────────────────
def _tukey_to_df(tukey):
    """Convert statsmodels TukeyHSD results to a tidy DataFrame."""
    if tukey is None:
        return None
    header, data = tukey._results_table.data[0], tukey._results_table.data[1:]
    return pd.DataFrame(data=data, columns=header)


# ──────────────────────────────────────────────────────────────────
# 3.  analyse_anova_by_location
# ──────────────────────────────────────────────────────────────────
def anova_by_location(df,
                              *,
                              include_likely=True,
                              post_hoc=True,
                              output_path=None):
    """
    Run the mouse-level ANOVA (and optional Tukey) for every
    location ≠ 'other', then write one Excel workbook.

    Returns
    -------
    results : dict  { location : {'anova': DataFrame,
                                  'tukey': DataFrame or None,
                                  'agg'  : DataFrame } }
    """
    results = {}
    for loc in [l for l in df['location'].unique() if l.lower() != 'other']:
        sub = df[df['location'] == loc]
        if sub.empty:
            continue
        out = anova(sub, post_hoc=post_hoc, include_likely=include_likely)
        out['tukey'] = _tukey_to_df(out['tukey'])  # convert now
        results[loc] = out

    # ---- Excel export --------------------------------------------------------
    if output_path is not None:
        with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:
            for loc, out in results.items():
                # Excel sheet names must be ≤31 chars and unique
                sheet_anova = (str(loc)[:25] + "_ANOVA")
                out['anova'].to_excel(writer, sheet_name=sheet_anova,
                                      startrow=0, index=True)

                if post_hoc and out['tukey'] is not None:
                    sheet_tukey = (str(loc)[:23] + "_Tukey")
                    out['tukey'].to_excel(writer, sheet_name=sheet_tukey,
                                          startrow=0, index=False)

        print(f"Excel written → {output_path}")

    return results


'''MIXED LOGISTIC REGRESSION'''

def run_mixed_logistic_by_location(df, output_path, posthoc_method='holm', alpha=0.05):
    """
    For each location != 'other', fit a Binomial GEE (marginal model with exchangeable correlation)
    clustering on 'mouse'.  Compute pairwise group‐vs‐group contrasts on the log‐odds scale,
    apply posthoc correction, and write one sheet per location in a single Excel workbook.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
          - 'group'           (categorical or string)
          - 'location'        (categorical or string)
          - 'mouse'           (categorical or string)
          - 'definitely alive'
          - 'definitely dead'
          - 'likely alive'
          - 'likely dead'
    output_path : str or pathlib.Path
        The full path for the output .xlsx; one sheet per location (excluding "other").
    posthoc_method : str, optional
        Method for multiple‐testing correction (default 'holm').
    alpha : float, optional
        Significance threshold (not used directly here, but could be added for filtering).
    """
    # 1) Filter out rows where location == 'other'
    df_filtered = df[df['location'] != 'other'].copy()
    if df_filtered.shape[0] == 0:
        raise ValueError("After dropping location == 'other', no rows remain.")
    
    # 2) Compute successes / failures / total
    df_filtered['successes'] = df_filtered['definitely alive'] + df_filtered['likely alive']
    df_filtered['failures']  = df_filtered['definitely dead'] + df_filtered['likely dead']
    df_filtered['total']     = df_filtered['successes'] + df_filtered['failures']
    
    # 3) Drop any rows where total == 0 (no cells counted)
    df_filtered = df_filtered[df_filtered['total'] > 0].copy()
    if df_filtered.shape[0] == 0:
        raise ValueError("All rows have total == 0 after combining counts.")
    
    # 4) Convert grouping columns to categorical
    df_filtered['group']    = df_filtered['group'].astype('category')
    df_filtered['location'] = df_filtered['location'].astype('category')
    df_filtered['mouse']    = df_filtered['mouse'].astype('category')
    
    # 5) Create an "observed proportion" column for GEE
    df_filtered['prop_alive'] = df_filtered['successes'] / df_filtered['total']
    
    # 6) Open an ExcelWriter so that we can write one sheet per location
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
        # 7) Loop over each valid location
        for loc in df_filtered['location'].cat.categories:
            df_loc = df_filtered[df_filtered['location'] == loc].copy()
            if df_loc.shape[0] == 0:
                continue
            
            # 8) Determine which groups actually appear at this location
            groups_present = sorted(df_loc['group'].astype(str).unique())
            if len(groups_present) < 2:
                # Need at least two groups to compare
                continue
            
            # 9) Build design matrix (exog) = constant + group dummies (drop_first=True)
            dummy_groups = pd.get_dummies(df_loc['group'].astype(str), drop_first=True)
            exog = sm.add_constant(dummy_groups).astype(float)
            
            # 10) Endog = observed proportion; weights = total count
            endog   = df_loc['prop_alive'].astype(float).values
            weights = df_loc['total'].astype(float).values
            
            # 11) Fit a Binomial GEE with Exchangeable correlation by 'mouse'
            try:
                gee_mod = GEE(endog,
                              exog,
                              groups=df_loc['mouse'],
                              family=Binomial(),
                              cov_struct=Exchangeable(),
                              weights=weights)
                gee_res = gee_mod.fit()
            except Exception as e:
                # If the GEE fails (e.g. perfect separation), skip this location
                print(f"Warning: GEE failed at location '{loc}': {e}")
                continue
            
            # 12) Extract parameter estimates and covariance matrix
            params_loc = gee_res.params       # Series indexed by ['const', <dummy names>...]
            cov_loc    = gee_res.cov_params() # DataFrame of covariance among coefficients
            
            # 13) Build pairwise contrasts among groups_present
            #     - baseline is the first group in alphabetical order
            baseline = groups_present[0]
            
            def coef_name_g(g):
                """
                Given a group name g, return the corresponding coefficient name:
                - If g == baseline, return None (its coefficient is implicitly 0).
                - Otherwise, return the dummy column name, which is exactly g (since get_dummies used the group label).
                """
                return g if (g != baseline) else None
            
            records = []
            for g1, g2 in itertools.combinations(groups_present, 2):
                name1 = coef_name_g(g1)
                name2 = coef_name_g(g2)
                
                if (name1 is not None) and (name2 is not None):
                    # both g1 and g2 are nonbaseline
                    est = params_loc[name1] - params_loc[name2]
                    var = (
                        cov_loc.loc[name1, name1]
                        + cov_loc.loc[name2, name2]
                        - 2 * cov_loc.loc[name1, name2]
                    )
                elif (name1 is not None) and (name2 is None):
                    # g2 is baseline
                    est = params_loc[name1]
                    var = cov_loc.loc[name1, name1]
                elif (name2 is not None) and (name1 is None):
                    # g1 is baseline
                    est = -params_loc[name2]
                    var = cov_loc.loc[name2, name2]
                else:
                    # both are baseline (should not happen if len(groups_present) >= 2)
                    continue
                
                se    = np.sqrt(var)
                z     = est / se
                p_raw = 2 * (1 - norm.cdf(abs(z)))
                ci_l  = est - 1.96 * se
                ci_h  = est + 1.96 * se
                
                records.append({
                    'group1':   g1,
                    'group2':   g2,
                    'estimate': est,
                    'std_err':  se,
                    'z_value':  z,
                    'p_value':  p_raw,
                    'ci_lower':  ci_l,
                    'ci_upper':  ci_h
                })
            
            comp_df_loc = pd.DataFrame(records)
            if comp_df_loc.empty:
                continue
            
            # 14) Apply Holm (or other) multiple‐testing correction on p-values
            _, p_adj, _, _ = multipletests(comp_df_loc['p_value'], method=posthoc_method)
            comp_df_loc['p_adj'] = p_adj
            
            # 15) Write this location’s results to its own sheet
            sheet_name = str(loc)[:31]  # Excel sheet names ≤31 chars
            comp_df_loc.to_excel(writer, sheet_name=sheet_name, index=False)
    
    return None