import numpy as np
import pandas as pd
import reed
import hilda_config as config
import pyreadstat


def check_against_expected(df1, m, release):
    problem = pd.read_csv("data/anna_compare.csv", usecols=['xwaveid', 'redudl_a', 'redudl_f'])
    df1['xwaveid'] = df1['xwaveid'].astype(int)
    sfx = config.release_suffix[release]
    df2, _ = pyreadstat.read_sav(f'data/part1/Combined {m}190{sfx}.sav')
    df2['xwaveid'] = df2['xwaveid'].astype(int)
    d10 = pd.merge(problem, df1, how='left', on='xwaveid')
    d20 = pd.merge(problem, df2, how='left', on='xwaveid')
    cols0 = reed.regex_select(d10.columns, ['^aedq\\d{3}', '^aedqunk$'])
    cols1 = reed.regex_select(d20.columns, ['^qedq\\d{3}', '^qedqunk$'])
    d20['final_quals'] = d20[cols1].sum(axis=1)
    d10['initial_quals'] = d10[cols0].sum(axis=1)
    p = pd.merge(d10, d20[['xwaveid', 'final_quals']], how='outer', on='xwaveid')
    p[['xwaveid', 'redudl_a', 'redudl_f', 'initial_quals', 'final_quals']+cols0]


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
    number_of_qual_cols = reed.regex_select(df.columns, [prefix+'edq\\d{3}', prefix+'edqunk$'])
    count = df[number_of_qual_cols].sum(axis=1, skipna=skipna)
    result = pd.DataFrame({id_col: df[id_col], count_name: count})
    return result


def compute_qualification_count_change(df1, df2, prefix1, prefix2, skipna=True):
    count_start = compute_qual_count(df1, prefix1, skipna)
    count_end = compute_qual_count(df2, prefix2, skipna)
    t = pd.merge(count_start, count_end, on=['xwaveid'], how='inner')
    t['quals_gained'] = t[f"{prefix2}_quals"] - t[f"{prefix1}_quals"]
    t['redudl'] = (t['quals_gained'] > 0).astype(int)
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


def compute_treatment_vars(df1, prefix1, prefix2, release):
    s, m = prefix1, prefix2

    # read the combined file for the the end treatment wave
    sfx = config.release_suffix[release]
    df2, _ = pyreadstat.read_sav(f'data/part1/Combined {m}190{sfx}.sav')

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


def compute_outcomes(df1, s, e, release):
    sfx = config.release_suffix[release]
    df3, _ = pyreadstat.read_sav(f'data/part1/Combined {e}190{sfx}.sav')

    start_wave_outcomes = compute_outcomes_at_wave(df1, s)
    final_wave_outcomes = compute_outcomes_at_wave(df3, e)
    del df3

    start_cols = [f'{s}{n[1:]}' for n in final_wave_outcomes]
    end_cols = [f'{e}{n[1:]}' for n in final_wave_outcomes]
    end_labels = [f'y_{n[1:]}' for n in final_wave_outcomes]
    diff_cols = [f'y_D{n[1:]}' for n in final_wave_outcomes]

    combined_outcomes = final_wave_outcomes.join(start_wave_outcomes)
    for start, end, diff in zip(start_cols, end_cols, diff_cols):
        combined_outcomes[diff] = combined_outcomes[end] - combined_outcomes[start]

    combined_outcomes.drop(columns=start_cols, inplace=True)
    combined_outcomes.rename(columns=dict(zip(end_cols, end_labels)), inplace=True)
    combined_outcomes.reset_index(inplace=True)
    return combined_outcomes


def compute_outcomes_at_wave(df, prefix):
    """
    Returns
    -----------

    """

    hrs_worked = f'{prefix}jbhruc'
    # note that we may introduce some unaccounted for uncertainty using an imputed value as a label
    wages = [f'{prefix}wsce', f'{prefix}wscei']
    mental_health = f'{prefix}ghmh'
    outcomes = [hrs_worked, mental_health]+wages
    dout = df[['xwaveid']+outcomes].copy()

    employment = f'{prefix}employment'
    employment_vals = df[f"{prefix}esdtl"].apply(simplify_employment)  # significantly missing
    dout[employment] = employment_vals

    dout.set_index('xwaveid', inplace=True)

    return dout
