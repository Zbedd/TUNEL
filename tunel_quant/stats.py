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
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import statsmodels.api as sm
from statsmodels.formula.api import glm
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

# def run_mixed_logistic_by_location(df, output_path, posthoc_method='holm', alpha=0.05):
#     """
#     For each location != 'other', fit a Binomial‐GLM (prop ~ C(group) + C(mouse)),
#     compute all pairwise group‐vs‐group contrasts (within that location),
#     apply post‐hoc p‐value correction, and write one Excel sheet per location.

#     Parameters
#     ----------
#     df : pandas.DataFrame
#         Must contain exactly these columns:
#           - 'name'
#           - 'group'           (categorical: e.g. 'A', 'B', 'C')
#           - 'location'        (categorical: e.g. 'brain', 'liver', etc., with possibly 'other')
#           - 'mouse'           (categorical ID for each mouse)
#           - 'definitely alive'
#           - 'definitely dead'
#           - 'likely alive'
#           - 'likely dead'

#     output_path : str or pathlib.Path
#         Path (including filename) for the output Excel workbook.  
#         One sheet per location (excluding location == 'other').

#     posthoc_method : str, optional
#         Any method accepted by statsmodels.stats.multitest.multipletests().
#         Default = 'holm'.

#     alpha : float, optional
#         Significance threshold (not directly used inside; provided for consistency).
#     """
#     # 1) Filter out rows where location == 'other'
#     df = df[df['location'] != 'other'].copy()
#     if df.shape[0] == 0:
#         raise ValueError("After dropping location == 'other', no rows remain.")

#     # 2) Compute successes / failures / total for every row
#     df['successes'] = df['definitely alive'] + df['likely alive']
#     df['failures'] = df['definitely dead'] + df['likely dead']
#     df['total'] = df['successes'] + df['failures']

#     # 3) Drop any rows where total == 0 (cannot fit binomial if denominator is zero)
#     df = df[df['total'] > 0].copy()
#     if df.shape[0] == 0:
#         raise ValueError("All rows have total == 0 after combining alive/dead counts.")

#     # 4) Ensure categorical dtypes
#     df['group'] = df['group'].astype('category')
#     df['location'] = df['location'].astype('category')
#     df['mouse'] = df['mouse'].astype('category')

#     # 5) We'll write one sheet per location. Prepare ExcelWriter now.
#     with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
#         # 6) Loop over each location (excluding 'other', already filtered out)
#         for loc in df['location'].cat.categories:
#             df_loc = df[df['location'] == loc].copy()
#             if df_loc.shape[0] == 0:
#                 # Shouldn't happen, but just in case
#                 continue

#             # 7) Identify how many distinct groups appear at this location
#             groups_loc = df_loc['group'].cat.categories.tolist()
#             # But some categories might have dropped out entirely at this location,
#             # so we re‐extract the actual level‐list present:
#             groups_present = sorted(df_loc['group'].unique())
#             if len(groups_present) < 2:
#                 # No pairwise possible if fewer than 2 groups
#                 # Skip this location
#                 continue

#             # 8) Compute proportion and set up the formula.
#             df_loc['prop'] = df_loc['successes'] / df_loc['total']
#             formula_loc = 'prop ~ C(group)'

#             # 9) Fit the Binomial GLM with weights = total
#             model_loc = glm(formula=formula_loc,
#                             data=df_loc,
#                             family=Binomial(),
#                             weights=df_loc['total'])
#             result_loc = model_loc.fit()

#             # 10) From the fitted result, extract params & covariance
#             params_loc = result_loc.params
#             cov_loc = result_loc.cov_params()

#             # Helper: how each group’s coefficient is named in params_loc
#             def coef_name_g(g):
#                 return f'C(group)[T.{g}]'

#             # 11) Build ALL pairwise contrasts among groups_present
#             records = []
#             for g1, g2 in itertools.combinations(groups_present, 2):
#                 # Determine if each is the baseline or has a coefficient
#                 # In statsmodels, the *first* category is taken as baseline
#                 # by default. We can re‐order categories if we want a different baseline.
#                 # Here, we'll assume whatever order pandas assigned is fine.
#                 name1 = coef_name_g(g1) if g1 != df_loc['group'].cat.categories[0] else None
#                 name2 = coef_name_g(g2) if g2 != df_loc['group'].cat.categories[0] else None

#                 # Compute estimate and variance of (β_g1 − β_g2)
#                 if name1 and name2:
#                     est = params_loc[name1] - params_loc[name2]
#                     var = (
#                         cov_loc.loc[name1, name1]
#                         + cov_loc.loc[name2, name2]
#                         - 2 * cov_loc.loc[name1, name2]
#                     )
#                 elif name1 and not name2:
#                     # g2 is baseline
#                     est = params_loc[name1]
#                     var = cov_loc.loc[name1, name1]
#                 elif name2 and not name1:
#                     # g1 is baseline → 0 − β_g2
#                     est = -params_loc[name2]
#                     var = cov_loc.loc[name2, name2]
#                 else:
#                     # both are baseline (impossible unless there's exactly one group)
#                     continue

#                 se = np.sqrt(var)
#                 z = est / se
#                 p_raw = 2 * (1 - norm.cdf(abs(z)))
#                 ci_low = est - 1.96 * se
#                 ci_high = est + 1.96 * se

#                 records.append({
#                     'group1': g1,
#                     'group2': g2,
#                     'estimate': est,
#                     'std_err': se,
#                     'z_value': z,
#                     'p_value': p_raw,
#                     'ci_lower': ci_low,
#                     'ci_upper': ci_high
#                 })

#             comp_df_loc = pd.DataFrame(records)
#             if comp_df_loc.empty:
#                 # No valid contrasts at this location
#                 continue

#             # 12) Adjust all p_values within this location
#             reject, p_adj, _, _ = multipletests(comp_df_loc['p_value'], method=posthoc_method)
#             comp_df_loc['p_adj'] = p_adj

#             # 13) Write this location’s results to its own sheet
#             sheet_name = str(loc)
#             if len(sheet_name) > 31:
#                 sheet_name = sheet_name[:31]

#             comp_df_loc.to_excel(writer, sheet_name=sheet_name, index=False)

#     # The function does not return a single "global" model,
#     # but you could return a dict of {location: fitted_result, ...} if desired.
#     return None


def run_mixed_logistic_by_location(df, output_path, posthoc_method='holm', alpha=0.05):
    """
    For each location != 'other', fit a two‐column Binomial GLM (successes, failures)
    with cluster‐robust standard errors (clustered on 'mouse'), compute pairwise
    group‐vs‐group contrasts, apply posthoc correction, and write one sheet per location.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
          - 'name'
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
        Method for multiple testing correction (default 'holm').

    alpha : float, optional
        Significance threshold (not used directly here).
    """
    # 1) Filter out rows where location == 'other'
    df_filtered = df[df['location'] != 'other'].copy()
    if df_filtered.shape[0] == 0:
        raise ValueError("After dropping location == 'other', no rows remain.")

    # 2) Compute successes / failures / total
    df_filtered['successes'] = df_filtered['definitely alive'] + df_filtered['likely alive']
    df_filtered['failures']  = df_filtered['definitely dead'] + df_filtered['likely dead']
    df_filtered['total']     = df_filtered['successes'] + df_filtered['failures']

    # 3) Drop any rows where total == 0
    df_filtered = df_filtered[df_filtered['total'] > 0].copy()
    if df_filtered.shape[0] == 0:
        raise ValueError("All rows have total == 0 after combining counts.")

    # 4) Ensure categorical dtypes for grouping columns
    df_filtered['group']    = df_filtered['group'].astype('category')
    df_filtered['location'] = df_filtered['location'].astype('category')
    df_filtered['mouse']    = df_filtered['mouse'].astype('category')

    # 5) Open an ExcelWriter so that we can write one sheet per location
    with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:

        # 6) Loop over each valid location
        for loc in df_filtered['location'].cat.categories:
            df_loc = df_filtered[df_filtered['location'] == loc].copy()
            if df_loc.shape[0] == 0:
                continue

            # 7) Determine which groups actually appear at this location
            #    Convert to strings to be safe
            groups_present = sorted(df_loc['group'].astype(str).unique())
            if len(groups_present) < 2:
                # Need at least two groups to compare
                continue

            # 8) Build two‐column endog for Binomial: [successes, failures]
            #    Make sure they are numeric (float) arrays
            successes = df_loc['successes'].astype(float).values
            failures  = df_loc['failures'].astype(float).values
            endog     = np.vstack([successes, failures]).T  # shape = (n_rows, 2)

            # 9) Build exog = constant + group‐dummies (drop_first=True)
            dummy_groups = pd.get_dummies(df_loc['group'].astype(str), drop_first=True)
            exog = sm.tools.tools.add_constant(dummy_groups).astype(float)

            # 10) Fit the GLM with cluster‐robust SEs on 'mouse'
            glm_mod = sm.GLM(
                endog,
                exog,
                family=sm.families.Binomial()
            )
            glm_res = glm_mod.fit(
                cov_type = "cluster",
                cov_kwds = {"groups": df_loc['mouse'].cat.codes},
                disp     = False
            )

            params_loc = glm_res.params    # Series indexed by ['const', <dummy names>...]
            cov_loc    = glm_res.cov_params()

            # 11) Build pairwise contrasts among groups_present
            #    - baseline is the first group in alphabetical order
            baseline = groups_present[0]

            #    - helper: name of the coefficient for group 'g' is exactly 'g'
            def coef_name_g(g):
                return g

            records = []
            for g1, g2 in itertools.combinations(groups_present, 2):
                name1 = coef_name_g(g1) if g1 != baseline else None
                name2 = coef_name_g(g2) if g2 != baseline else None

                if name1 and name2:
                    est = params_loc[name1] - params_loc[name2]
                    var = (
                        cov_loc.loc[name1, name1]
                        + cov_loc.loc[name2, name2]
                        - 2 * cov_loc.loc[name1, name2]
                    )
                elif name1 and not name2:
                    # g2 is baseline, so its coefficient is 0
                    est = params_loc[name1]
                    var = cov_loc.loc[name1, name1]
                elif name2 and not name1:
                    # g1 is baseline → 0 − params_loc[name2]
                    est = -params_loc[name2]
                    var = cov_loc.loc[name2, name2]
                else:
                    # both are baseline (impossible unless there's exactly one group)
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
                    'ci_lower': ci_l,
                    'ci_upper': ci_h
                })

            comp_df_loc = pd.DataFrame(records)
            if comp_df_loc.empty:
                continue

            # 12) Apply Holm (or other) multiple‐testing correction
            _, p_adj, _, _ = multipletests(comp_df_loc['p_value'], method=posthoc_method)
            comp_df_loc['p_adj'] = p_adj

            # 13) Write this location’s results to its own sheet
            sheet_name = str(loc)[:31]  # Excel sheets must be ≤31 chars
            comp_df_loc.to_excel(writer, sheet_name=sheet_name, index=False)

    return None