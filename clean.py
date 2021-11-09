import pandas as pd


def add_list_to_dict(lst, dictionary, value):
    """Add all the values to the directory with the same fixed value."""
    for v in lst:
        dictionary[v] = value


def drop_mostly_missing_columns(df, threshold=0.99):
    """Drops columns (inplace) where more than threshold proportion of values are nan"""
    mostly_missing = df.columns[df.isnull().mean(axis=0) > threshold]
    df.drop(columns=mostly_missing, inplace=True)
    print(f"Dropping {len(mostly_missing)} columns with more than {threshold*100:.0f}% missing ")
    return mostly_missing


def drop_constant_columns(df):
    """Drop columns with standard deviation zero."""
    std = df.std()
    constant_variables = list(std[std == 0].index)
    df.drop(columns=constant_variables, inplace=True)
    print(f"Dropping {len(constant_variables)} columns that are constant or entirely missing")
    return constant_variables



def create_interaction_columns(df, columnsA, columnsB):
    """Create columns containing the interaction of each column in columnsA with each column in columnsB"""
    df_out = df.copy()
    assert len(set(columnsA).intersection(columnsB)) == 0, "No columns should be in both A and B"
    for i in columnsA:
        for j in columnsB:
            df_out[f'interact_{i}_{j}'] = df[i]*df[j]
    return df_out
