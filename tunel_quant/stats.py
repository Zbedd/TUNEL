'''
All statistics functions are built to accept the output of summarize_analysis,
which has the following columns:
        ['name', 'group', 'location', 'mouse', 
        'definitely alive', 'definitely dead', 'likely alive', 'likely dead']

'''

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


def mixed_logistic(df, *, include_likely=True, level=0.95, summary=True, fit_method="vb"):
    """
    Hierarchical logistic regression (dead/ alive) with random intercepts
    for mouse and image.  Accepts the *image-level* count table and
    explodes it internally.

    Parameters
    ----------
    df : DataFrame  – columns:
         ['name','group','location','mouse',
          'definitely alive','definitely dead',
          'likely alive','likely dead']
    include_likely : bool  – add 'likely' counts when True.
    summary        : bool  – print coefficient table.
    fit_method     : {"vb","map"} – Variational Bayes (fast) or Laplace MAP.

    Returns
    -------
    result : BinomialBayesMixedGLMResults
    """
    
    import numpy as np
    import pandas as pd
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
    from scipy.stats import norm
    
    def _explode(r):
        alive = r['definitely alive'] + (r['likely alive'] if include_likely else 0)
        dead  = r['definitely dead']  + (r['likely dead']  if include_likely else 0)
        return pd.DataFrame({
            'dead':  np.r_[np.zeros(alive, int), np.ones(dead, int)],
            'group': r['group'],
            'mouse': r['mouse'],
            'image': r['name']
        })
    nuclei = pd.concat((_explode(r) for _, r in df.iterrows()), ignore_index=True)

    vc = {"mouse": "0 + C(mouse)", "image": "0 + C(image)"}
    mdl = BinomialBayesMixedGLM.from_formula("dead ~ C(group)", vc, nuclei)
    res = mdl.fit_vb()                          # Bayes VB ⇒ no conf_int()

    if summary:
        print(res.summary())                    # full table
        # DIY 95 % credible intervals
        z = norm.ppf(0.5 + level/2)
        ci = pd.DataFrame({
            'OR':   np.exp(res.fe_mean),
            'low':  np.exp(res.fe_mean - z*res.fe_sd),
            'high': np.exp(res.fe_mean + z*res.fe_sd)
        }, index=mdl.data.param_names)
        print(f"\nPosterior {int(level*100)} % credible intervals (odds-ratios)")
        print(ci)

    return res