import numpy as np
import pandas as pd
import reed
import pyreadstat


def compute_qual_count(df, prefix, skipna):
    """
    Parameters
    -----------
    df: pd.DataFrame
        the cross-sectional data for a given wave
    prefix: str
        the letter prefix corresponding to that wave
    skipna: bool
        when computing the sum of qualifications for an individual, should na be skipped
        or should the sum be na if one is missing?

    Returns
    -----------
    result: pd.DataFrame
        a two column table with recording the id and number of qualifications for each individual in that wave.
    """
    id_col = 'xwaveid'
    count_name = f'{prefix}_quals'
    number_of_qual_cols = reed.regex_select(df.columns, prefix+'edq\d{3}')
    count = df[number_of_qual_cols].sum(axis=1, skipna=skipna)
    result = pd.DataFrame({id_col: df[id_col], count_name: count})
    return result


def compute_qualification_count_change(df1, df2, prefix1, prefix2, skipna=True):
    count_start = compute_qual_count(df1, prefix1, skipna)
    count_end = compute_qual_count(df2, prefix2, skipna)
    t = pd.merge(count_start, count_end, on=['xwaveid'], how='inner')
    t['quals_gained'] = t[f"{prefix2}_quals"] - t[f"{prefix1}_quals"]
    t['redudl'] = (t['quals_gained'] > 1).astype(int)
    result = t[['xwaveid', 'redudl']]
    return result


def compute_highest_level_education_change(df1, df2, prefix1, prefix2):
    hed_var = 'edhigh1'  # lowest value => most educated, 10 or < 0 => undetermined
    var1 = prefix1 + hed_var
    var2 = prefix2 + hed_var
    d1 = df1[['xwaveid', var1]].copy()
    d2 = df2[['xwaveid', var2]].copy()
    d1.loc[(d1[var1] < 0) | (d1[var1] > 9), var1] = np.nan  # set undefined values to nan
    d2.loc[(d2[var2] < 0) | (d2[var2] > 9), var2] = np.nan
    d = pd.merge(d1, d2, on=['xwaveid'], how='inner')  # we only keep ids observed in both waves
    # will be positive if increased since lower => better
    d['qual_level_change'] = d[var1] - d[var2]
    d['reduhl'] = (d['qual_level_change'] > 0).astype(int)
    result = d[['xwaveid', 'reduhl']].copy()
    return result


def compute_confusion(v1, v2, label1, label2):
    assert len(v1) == len(v2), "value arrays must be the same length"
    t00 = ((v1 == 0) & (v2 == 0)).sum()
    t01 = ((v1 == 0) & (v2 == 1)).sum()
    t10 = ((v1 == 1) & (v2 == 0)).sum()
    t11 = ((v1 == 1) & (v2 == 1)).sum()
    matrix = [[t00, t01], [t10, t11]]
    col_names = [f"{label2}==0", f"{label2}==1"]
    row_names = [f"{label1}==0", f"{label1}==1"]
    return pd.DataFrame(matrix, columns=col_names, index=row_names)


def compute_treatment_vars(df1, prefix1, prefix2):
    s, m = prefix1, prefix2

    # read the combined file for the the end treatment wave
    df2, _ = pyreadstat.read_sav(f'../part1/Combined {m}190c.sav')

    # compute treatment
    t1 = compute_qualification_count_change(df1, df2, s, m)
    t2 = compute_highest_level_education_change(df1, df2, s, m)
    t = pd.merge(t1, t2, on='xwaveid', how='outer')
    t['redufl'] = ((t['redudl'] == 1) | (t['reduhl'] == 1)).astype(int)
    del df2
    return t


def simplify_employment(v):
    """
    Compute empoyment category based on the `_esdtl` variable.

    Simplify down to: employed FT (1), employed PT (2), not in paid employment (3)
    """
    if v < 0:
        return np.nan  # missing
    if v > 2:
        return 3
    return v


def compute_outcomes(prefix):
    df3, _ = pyreadstat.read_sav(f'../part1/Combined {prefix}190c.sav')

    hrs_worked = f'{prefix}jbhruc'
    # note that we may introduce some unaccounted for uncertainty using an imputed value as a label
    wages = [f'{prefix}wsce', f'{prefix}wscei']
    mental_health = f'{prefix}ghmh'
    employment = f'{prefix}employment'
    df3[employment] = df3[f"{prefix}esdtl"].apply(simplify_employment)  # significantly missing
    outcomes = [hrs_worked, mental_health, employment]+wages
    outnames = [f"y_{s[1:]}" for s in outcomes]
    dout = df3[['xwaveid']+outcomes].copy()
    dout.columns = ['xwaveid']+outnames
    del df3
    return dout
