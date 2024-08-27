import pandas as pd

from storage import TableWriteOption, get_last_updated_date


def get_table_write_option(incremental: bool) -> TableWriteOption:
    return TableWriteOption.APPEND if incremental else TableWriteOption.REPLACE


# Converts all numeric values in a df to numeric, while leaving non-numeric values as is.
def graceful_df_to_numeric(df):
    # Create a new df that converts all columns to numeric, filling with NaN for non-numeric values
    df_numeric = df.apply(lambda col: pd.to_numeric(col, errors='coerce'))
    # Combine the numeric df with the original df, filling in any NaN values in the numeric df with the original values.
    return df_numeric.combine_first(df)


# Used when running incremental updates, this function filters out rows before (and equal to)
# the most recent date in the table.
def drop_existing_rows(df, table_name: str, date_col: str, symbol: str = ''):
    last_updated_date = get_last_updated_date(table_name, date_col, symbol)
    return df[df.index > last_updated_date] if last_updated_date else df
