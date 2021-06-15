import pandas as pd
import karmahutils as kut
import re

version_number = '1.4.2'
version = 'moonshade'


def parse_columns_date(
        data,
        columns,
        errors='raise',
        first_day=False,
        first_year=False,
        utc=None,
        format=None,
        exact=True,
        unit=None,
        infer_datetime_format=False,
        origin='unix',
        cache=True,
        inplace=False,
        silent_mode=True
):
    df = data if inplace else data.copy()
    if not silent_mode:
        kut.display_message('parsing dates')
    for column in columns:
        if not silent_mode:
            kut.display_message(column, secondary=True)
            print(df[column].value_counts(dropna=False).head())
            print('type', df[column].dtype)
        if 'datetime' in str(data[column].dtype):
            continue
        data[column] = pd.to_datetime(
            data[column],
            errors=errors,
            dayfirst=first_day,
            yearfirst=first_year,
            utc=utc,
            format=format,
            exact=exact,
            unit=unit,
            infer_datetime_format=infer_datetime_format,
            origin=origin,
            cache=cache,
        )
        if not silent_mode:
            print(df[column].value_counts(dropna=False).head())
            print('type', df[column].dtype)

        return None if inplace else df


def fix_literal_numbers(value, thousand_separators=None, decimal_separator='.', unit=None):
    if thousand_separators is None:
        thousand_separators = [' ', ',']
    if pd.isnull(value):
        return
    if kut.is_numeric(value):
        return value
    thousand_separators = [thousand_separators] if type(thousand_separators) != list else thousand_separators
    if unit:
        value.replace(unit, '')
    if any(thousand_separators):
        for separator in thousand_separators:
            value = value.replace(separator, '')
    value = value.replace(decimal_separator, '.')
    value = re.sub(r'[^\d,^.]', '', value)
    return float(value) if '.' in value else int(value)


def fix_columns_literal_numbers(
        data,
        columns,
        thousand_separators=None,
        decimal_separators='.',
        silent_mode=True,
        inplace=False,
        units=None
):
    if thousand_separators is None:
        thousand_separators = [' ', ',']
    df = data if inplace else data.copy()
    if not silent_mode:
        kut.display_message('fixing literal numbers')
    for column in columns:
        this_unit = units.get(column, '') if units is not None else units
        new_column = '_'.join([column, this_unit]) if this_unit else column
        if not silent_mode:
            kut.display_message(column, secondary=True)
            print(df[column].value_counts(dropna=False).head())
            print('type', df[column].dtype)
        this_thousand_separators = thousand_separators.get(column, 'true') if type(
            thousand_separators) == dict else thousand_separators
        this_decimal_separator = decimal_separators.get(column, None) if type(
            decimal_separators) == dict else decimal_separators
        df[column] = df[column].apply(lambda x: fix_literal_numbers(
            x,
            thousand_separators=this_thousand_separators,
            decimal_separator=this_decimal_separator,
            unit=this_unit
        )
                                      )
        df.rename(columns={column: new_column}, inplace=True)
        if not silent_mode:
            print(df[new_column].value_counts(dropna=False).head())
            print('type', df[new_column].dtype)
    return None if inplace else df


def inventory_data(dataframe, sample_size=3):
    drop_candidates = []
    contains_null = []
    for column in dataframe.columns:
        print('analyzing', column)
        data = dataframe[column]
        print(data.dtype)
        nb_uniques = len(data.unique())
        nb_null = data.isnull().sum()
        print(nb_uniques, 'unique values')
        if nb_null:
            contains_null.append(column)
            print(nb_null, 'null values', round(nb_null * 100 / len(data), 2), '%')
        if nb_uniques < 10:
            print(data.value_counts(dropna=True))
            drop_candidates.append(column)
        else:
            print('example values')
            print(data.value_counts().sample(sample_size))
        print('_' * 7)
    return {
        'drop_candidates': drop_candidates,
        'contains_null': contains_null
    }


def check_inventory(
        data,
        inventoried=None,
        ok=None,
        drop_columns=None,
        literals_numeric=None,
        literals_boolean=None,
        dates_to_parse=None,
        return_status=False
):
    ok = [] if ok is None else ok
    drop_columns = [] if drop_columns is None else drop_columns
    literals_numeric = [] if literals_numeric is None else literals_numeric
    literals_boolean = [] if literals_boolean is None else literals_boolean
    dates_to_parse = [] if dates_to_parse is None else dates_to_parse
    inventoried = \
        ok + dates_to_parse + drop_columns + literals_numeric + literals_boolean if inventoried is None else inventoried
    completion = len(inventoried) - len(data.columns)
    missing = [X for X in data.columns if X not in inventoried]
    overflow = [X for X in inventoried if X not in data.columns]
    print('completion', completion)
    print('missed in inventory')
    print(missing)
    print('inventoried by mistake (overflow)')
    print(overflow)
    return {
        'completion': completion,
        'missing': missing,
        'overflow': overflow
    } if return_status else None


def fix_literal_boolean(value, true_value='true', false_value=None):
    if type(value) == bool:
        return value
    if pd.isnull(value):
        return
    if false_value is not None:
        return value.lower() != false_value.lower()
    return value.lower() == true_value.lower()


def fix_columns_literal_boolean(data, columns, true_value='true', false_value=None, silent_mode=True, inplace=False):
    df = data if inplace else data.copy()
    if not silent_mode:
        kut.display_message('fixing literal booleans')
    for column in columns:
        if not silent_mode:
            kut.display_message(column, secondary=True)
            print(df[column].value_counts(dropna=False))
            print('type', df[column].dtype)
        this_true_value = true_value.get(column, 'true') if type(true_value) == dict else true_value
        this_false_value = false_value.get(column, None) if type(false_value) == dict else false_value
        df[column] = df[column].apply(lambda x: fix_literal_boolean(x, this_true_value, this_false_value))
        if not silent_mode:
            print(df[column].value_counts(dropna=False))
            print('type', df[column].dtype)
    return None if inplace else df


print('loaded karmah inventory utils', version, version_number)
