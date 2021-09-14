import copy
import datetime
import time
import dataiku
import numpy as np
import pandas as pd
from random import randint, choice, random
import string
import re
from socket import inet_ntoa
from struct import pack
import warnings
from sklearn import preprocessing
import itertools

warnings.simplefilter(action='ignore', category=FutureWarning)
version_info = "v1.18"
version_type = 'KAP Code Twitter'
authors = ['Yann Girard', 'Bastien Clausier', 'Afaf Ghazi']
contact = 'yann.girard@gmail.com'
lib_name = 'karmahutils'
purpose = """This lib contains most of the useful quality of life functions
used in my everyday coding life.It used to be the standalone of the karmah function series.
It has been shared in various projects I have been involved in."""


def split_me(data, batch_size=2):
    return [df for g, df in data.groupby(np.arange(len(data)) // batch_size)]


def load_info(library_name=lib_name, build=version_info):
    print(f'loaded {library_name} ,version : {version_type} ,build: {build}')


# noinspection PyShadowingNames
def version_info(library_name=lib_name, build=version_info, purpose=purpose, authors=authors, contact=contact):
    print(f"{library_name} ,version : {version_type} ,build: {build}")
    print('_' * 5)
    print(purpose)
    print('_' * 5)
    print(f'contributors {authors}')
    print(f'contact:{contact}')


# source: https://www.kaggle.com/ratan123/m5-forecasting-lightgbm-with-timeseries-splits
def reduce_mem_usage(df, verbose=True):
    """borrowed from https://www.kaggle.com/ratan123/m5-forecasting-lightgbm-with-timeseries-splits"""
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def numeric_columns(df, include_dates=True):
    """ provide a list of columns that are float64,int64 or datetime64"""
    scope = ['int64', 'float64']
    if include_dates:
        scope.append('datetime64[ns]')
    return [X for X in df.columns if str(df[X].dtypes) in scope]


def pairs(a):
    """creates pairs of elements of a single list (no redundancies)"""
    return [X for X in list(itertools.product(a, a)) if X[0] != X[1]]


def normalize_df(df):
    """normalize a numeric df (no type control implemented yet) """
    x = df.values
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    scaled_df = pd.DataFrame(x_scaled)
    renaming_rules = dict(zip(scaled_df.columns, df.columns))
    scaled_df.rename(columns=renaming_rules, inplace=True)
    return scaled_df


def split_data_from_column(data, column_name):
    return dict(tuple(data.groupby(column_name)))


def str_is_number(s):
    try:
        complex(s)  # for int, long, float and complex
    except ValueError:
        return False

    return True


def audit_null_entries(dataframe, column, print_only=True):
    null_values = len(dataframe[pd.isnull(dataframe[column])])
    total = len(dataframe)
    ratio = null_values / total
    print(null_values, '/', total, str(round(ratio * 100, 2)) + '%')
    return {"total": total, 'null': null_values, 'ratio': ratio} if not print_only else None


def substitute_entry(dict_list, new_entry, criteria, allowed_addition=False):
    """substitute a dic entry in a _dict_list_ with *new_entry* using *criteria* key as an index
    if *allowed_addition* the new entry will be added if *criteria* value does not exists"""
    entry_exists = len([X for X in dict_list if X[criteria] == new_entry[criteria]]) > 0
    print('entry exists', entry_exists, 'allowed addition', allowed_addition)
    if allowed_addition and not entry_exists:
        print('added new entry')
        return dict_list + [new_entry]
    else:
        print('updating' if entry_exists else 'addition forbidden')
        return [new_entry if X[criteria] == new_entry[criteria] else X for X in dict_list]


def random_ip():
    return inet_ntoa(pack('>I', randint(1, 0xffffffff)))


def reformat_float_as_str(id_value):
    """ QOL function used to reformat numeric id columns containing empty values"""
    if pd.isnull(id_value):
        return ''
    return str(int(id_value))


def list_str_to_list(list_str, separator=','):
    """to revert the results from unique_values_as_string()"""
    list_str = list_str.replace('[', '').replace(']', '')
    return list_str.split(separator)


def unique_values_as_string(data=None, column_name=None, array=None, separator=',', list_shaped=False, unique=True):
    """return unique values of a column (data[column_name]) or list(array)  as a (separator)separated-string
    list_shape is a QOL arg that adds '[]' and forces ',' as separators to the string (for a future eval) """
    value = None
    if array is not None:
        value = np.unique(array) if unique else array
    if data is not None:
        value = data[column_name].unique() if unique else data[column_name].tolist()
    if value is None:
        raise Exception('you must pass a list or series of input values to kut.unique_values_as_string()')
    bracket_open = '[' if list_shaped else ''
    bracket_closed = ']' if list_shaped else ''
    separator = ',' if list_shaped else separator
    return bracket_open + separator.join([str(X) for X in value]) + bracket_closed


def substitute_elements_in_list(a_list, substitution_rule):
    """replace elements in list using a substitution rule defined in a dict (same as rename in pd)
    use with a list of same type elements.
    warning: the use of np.array will set the resulting list elements type to the most inclusive dtypes"""
    np_array_list = np.array(a_list)
    for element in substitution_rule.keys():
        np_array_list = np.where(np_array_list == element, substitution_rule[element], np_array_list)
    return list(np_array_list)


def get_unique_values(dict_array, key):
    return list(set([X[key] for X in dict_array.keys()]))


def convert_timestamp_to_datetime(timestamp):
    return datetime.datetime.fromtimestamp(timestamp / 1000) if not pd.isnull(timestamp) else pd.NaT


def assign_numeric_dtypes(dataframe):
    for column in dataframe.columns:
        if dataframe[column].dtype != 'O':
            continue
        if dataframe[column].str.isdecimal().all():
            dataframe[column] = dataframe[column].astype(np.int64)
            continue
        try:
            dataframe[column] = dataframe[column].astype(np.float64)
            continue
        except (TypeError, ValueError):
            continue
    return dataframe


def is_valid_date(date, date_format='%Y-%m-%d %H:%M:%S'):
    try:
        datetime.datetime.strptime(date, date_format)
        return True
    except ValueError:
        return False


def reformat_dict(dictionary, date_format='%Y-%m-%d %H:%M:%S'):
    for field in dictionary.keys():
        if dictionary[field] in ['True', 'False']:
            dictionary[field] = eval(dictionary[field])
            continue
        if not pd.isnull(re.match(r'^-?\d+(?:\.\d+)?$', dictionary[field])):
            dictionary[field] = eval(dictionary[field])
            continue
        if is_valid_date(date=dictionary[field], date_format=date_format):
            dictionary[field] = datetime.datetime.strptime(dictionary[field], date_format)
            continue
    return dictionary


def audit_for_null_values(df, identifier_columns, assessment_columns=None):
    if assessment_columns is None:
        assessment_columns = []
    start = yet()
    print(nu(), 'starting audit_for_null_values()')
    null_values = []
    for i, row in df.iterrows():
        for cell in row.iteritems():
            if pd.isnull(cell[1]):
                entry = {identifier: row[identifier] for identifier in identifier_columns}
                entry['empty_field'] = cell[0]
                null_values.append(entry)
    job_done(start=start, additions='computing audit')
    df_null = filter_on_values(
        df,
        value_list=[X[identifier_columns[0]] for X in null_values],
        column_name=identifier_columns[0]
    )
    if len(assessment_columns) > 0:
        display_message('assessing distributions')
        for assess_column in assessment_columns:
            display_message(assess_column)
            print(df_null[assess_column].value_counts())
    return {'audit': pd.DataFrame(null_values), 'data': df_null}


def filter_on_values(df, value_list, column_name):
    original_index = df.index.name
    if not pd.isnull(original_index):
        df.reset_index(inplace=True)
    out = df.merge(pd.DataFrame(value_list, columns=[column_name]), on=[column_name])
    if not pd.isnull(original_index):
        df.set_index(original_index, inplace=True)
        out.set_index(original_index, inplace=True)
    return out


def write_with_compliance(data_frame=None, dataframe=None, dataset_name=None, dataset=None, filters=None):
    """a container for dataiku write_with_schema() applying every compliance constraints on dataset schema.
    dataset is a dataiku dataset object
    data_frame is a dataframe object to be written
    alternatively to the dataset option you can pass a dataset name in dataset_name argument"""

    if dataframe is None:
        # this step is here for retro-compat and should be removed asap
        dataframe = data_frame
    if filters:
        other_columns = [X for X in dataframe.columns if X not in filters]
        other_columns.sort()
        dataframe = dataframe[filters + other_columns]
    else:
        dataframe.sort_index(axis=1, inplace=True)
    if dataset_name:
        dataset = dataiku.Dataset(dataset_name)
    if not (dataset_name or dataset):
        print("no output dataset provided in karmahutils.write_with_compliance()")
        raise Exception("no output dataset provided")
    dataset.write_with_schema(dataframe)
    return 1


def random_character():
    return choice(string.ascii_letters)


def random_string(length=4):
    string_random = random_character()
    for i in range(length - 1):
        string_random += random_character()
    return string_random


def random_df(nb_rows=7):
    out = []
    random_values = ['value' + str(i) for i in range(1, 4)]
    for i in range(nb_rows):
        out.append({
            'intValue': randint(0, 10),
            'discreteValue': random_string(),
            'percentage': round(random(), 2) * 100,
            'Boolean': bool(randint(0, 1)),
            'Ip': random_ip(),
            'discreteValues': random_values[randint(0, len(random_values) - 1)]
        })
    return pd.DataFrame(out)


def money(amount, currency='€', cents_offset=2):
    """
    float amount: amount to display
    """
    return str(round(amount, cents_offset)) + ' ' + currency


def to_camelcase(a_string):
    if len(a_string.split(' ')) == 1:
        return a_string.lower()
    else:
        string_words = a_string.split(' ')
        return ''.join([string_words[0].lower()] + [X[0].upper() + X[1:].lower() for X in string_words[1:]])


def display_message(message, other_lines=None, line_sep='*', side_sep='*', secondary=False, no_return=True):
    if secondary:
        line_sep = '-'
        side_sep = ''

    if not type(message) == list:
        message = [message]
    if not other_lines:
        other_lines = []
    if not type(other_lines) == list:
        other_lines = [[other_lines]]
    if (type(other_lines) == list) and (any([type(X) == str for X in other_lines])):
        other_lines = [other_lines]

    message = [message] + other_lines
    max_length = max([len(X) - 1 + sum(len(str(Y)) for Y in X) for X in message])

    # Printing
    print(line_sep * (max_length + 4))

    for message_line in message:
        line_length = sum([len(str(X)) for X in message_line]) + len(message_line) - 1
        diff_length = max_length - line_length
        line = side_sep + " "
        for messagePart in message_line:
            line += str(messagePart) + ' '
        line += " " * diff_length + side_sep
        print(line)
    print(line_sep * (max_length + 4))

    if no_return:
        return
    else:
        return message


def dss2datetime(dss_date):
    dss_components = dss_date.replace('Z', '').split('T')
    dss_day_components = dss_components[0].split('-')
    dss_hour_components = dss_components[1].split(':')
    all_components = dss_day_components + dss_hour_components
    return datetime.datetime(*map(int, all_components))


def time_col_dss_to_datetime(df, time_column_list=None):
    if time_column_list is None:
        time_column_list = ['createdAt', 'lastActivityAt', 'lastUpdateAt', 'sendingDate']
    for col in [X for X in df.columns if X in time_column_list]:
        df[col] = df[col].apply(lambda x: dss2datetime(x))
    return df


def job_done(start, additions='', no_return=True):
    print('job done in', str((yet() - start)), additions)
    if no_return:
        return
    else:
        return 'job done in ' + str((yet() - start)) + ' ' + additions


def column_names_in_camelcase(df, id_prefix=''):
    new_col_name = {}
    for col in df.columns:
        new_col_name[col] = col[0].lower() + col[1:] if col != 'ID' else id_prefix + 'ID'
    return df.rename(columns=new_col_name)


def load_dataset(
        dataset_name,
        columns=None,
        sampling='head',
        sampling_column=None,
        limit=None,
        ratio=None,
        infer_with_pandas=True,
        parse_dates=True,
        bool_as_str=False,
        float_precision=None,
        conversion_dwh_di=False
):
    start = datetime.datetime.now()
    print(str(start), "loading", dataset_name)
    df = dataiku.Dataset(dataset_name).get_dataframe(columns=columns, sampling=sampling,
                                                     sampling_column=sampling_column, limit=limit, ratio=ratio,
                                                     infer_with_pandas=infer_with_pandas,
                                                     parse_dates=parse_dates, bool_as_str=bool_as_str,
                                                     float_precision=float_precision)
    print("df", dataset_name, "loaded:", len(df), 'in', str(datetime.datetime.now() - start))
    return rename_columns(df) if conversion_dwh_di else df


def true_copy(df):
    df3 = df.copy(deep=True)
    for c in df3:
        df3[c] = [copy.deepcopy(e) for e in df3[c]]
    return df3


def yet():
    return datetime.datetime.now()


def nu(short=False):
    return str(datetime.datetime.now()) if not short else str(datetime.datetime.now()).split(' ')[0]


def is_numeric(value, exclude_null=False):
    """ return true for None,NaN and any subdtype of int and float.
     exclude_null argument avoid counting pd.isnull(value) as numeric"""
    if exclude_null and pd.isnull(value):
        return False
    return np.issubdtype(int, type(value)) or np.issubdtype(float, type(value))


def back_to_int(new_df, first_joinee, second_joinee, float_fill=0.0, int_fill=0):
    """new_df as (usually) the result of an outer join where some values are empty
    first_joinee,second_joinee are the original dataset who did contain those numeric values
    this function (in this form) only works if all numeric columns originate from the same dataframe.
    """
    for old_df in [first_joinee, second_joinee]:
        int_col = [x for x in new_df.columns if x in old_df.columns and np.issubdtype(int, old_df.dtypes.loc[x])]
        float_col = [x for x in new_df.columns if x in old_df.columns and np.issubdtype(float, old_df.dtypes.loc[x])]
        for col in int_col:
            new_df[col] = new_df[col].fillna(int_fill).astype(int)
        for col in float_col:
            new_df[col] = new_df[col].fillna(float_fill).astype(float)
    return new_df


def from_timestamp(timestamp, unit='ms'):
    modifier = 1
    if unit == 'ms':
        modifier = 1000
    return datetime.datetime.fromtimestamp(timestamp / modifier)


def to_timestamp(date, unit='ms'):
    modifier = 1
    if unit == 'ms':
        modifier = 1000
    return int(time.mktime(date.timetuple()) * modifier)


def file_stamp(short=False):
    Y = str(datetime.datetime.now().year)
    M = str(datetime.datetime.now().month).zfill(2)
    D = str(datetime.datetime.now().day).zfill(2)
    H = str(datetime.datetime.now().hour - 1).zfill(2)
    m = str(datetime.datetime.now().minute).zfill(2)
    return Y + "-" + M + "-" + D if short else Y + "-" + M + "-" + D + "_" + H + m


def same_day(date1, date2):
    return date1.strftime('%Y-%m-%d') == date2.strftime('%Y-%m-%d')


def update(adf, bdf):
    """ update dataframe with values from dataframe b that are not in a."""
    adf = true_copy(adf)
    bdf = true_copy(bdf)
    adf_only_col = [X for X in adf.columns if X not in bdf.columns]
    for col in adf_only_col:
        bdf[col] = pd.Series(None, index=bdf.index)
    adf.update(bdf)
    return adf.append(bdf.loc[bdf.index.difference(adf.index)])


def dtypes_to_mailjet_types(col, df):
    matches = ['int', 'float', 'date']
    column_type = str(df[col].dtypes)
    for match in matches:
        if match in column_type:
            return match
        else:
            continue
    return 'str'


def audit_join(df_left, df_right):
    display_message('analyzing columns')
    out = {}
    display_message('overlap', secondary=True)
    in_overlap = df_left.columns.intersection(df_right.columns)
    print(in_overlap)
    if in_overlap.tolist():
        print('some columns are in overlap , consider using the lsuffix or rsuffix option in your join')
    out['columnsOverlap'] = bool(in_overlap.tolist())
    display_message('only on the left side', secondary=True)
    only_left = df_left.columns.difference(df_right.columns)
    print(only_left)

    display_message('only right', secondary=True)
    only_right = df_right.columns.difference(df_left.columns)
    print(only_right)

    display_message('row index(es)')
    print('index_left')
    print(df_left.index.names)
    print('index_right')
    print(df_right.index.names)

    if df_left.index.names != df_right.index.names:
        if all([len(X.index.names) == 1 for X in [df_left, df_right]]):
            print('index(es) are different but join should work as they are both unidimensional')
            out['compatibleRowIndexes'] = True
            return out
        if len(df_left.index.names) != len(df_right.index.names):
            print('row indexes have different dimensions , join should not work')
            out['compatibleRowIndexes'] = False
            return out
        print('you have multi dimensional indexes of different names, join will not work')
        out['compatibleRowIndexes'] = False
        return out
    else:
        if all([len(X.index.names) == 1 for X in [df_left, df_right]]):
            print('trivial case of unidimensional identical index(es) , join should work')
            out['compatibleRowIndexes'] = True
            return out
        print('multi-dimensional indexes of the same name , join should work')
        out['compatibleRowIndexes'] = True
        return out


def audited_join(df_left, df_right, on=None, how='left', lsuffix='', rsuffix='', sort=False):
    audit_result = audit_join(df_left, df_right)
    if not audit_result['compatibleRowIndexes']:
        raise Exception('impossible to make the join due to incompatible row indexes')
    if audit_result['columnsOverlap'] and (not any([bool(lsuffix), bool(rsuffix)])):
        raise Exception("impossible to join with columns in overlap if you don't use lsuffix or rsuffix option")
    return df_left.join(df_right, on=on, how=how, lsuffix=lsuffix, rsuffix=rsuffix, sort=sort)


def join_with_default_value(df_left, df_right, right_column, how='left', lsuffix='', rsuffix='', lindex=None,
                            rindex=None, sort=False, audited=False, default_value='others'):
    """
    Function: This function has been devised to perform enrichment from a referential table containing a 'others' key
    with a corresponding default value. However we kept the names of the variables generic to this functional first
    use.

    Suppose you are to enrich some orders with endUserGeozone or a product series. df_left will be your orders ,
    on the right the referential table. right_column will be 'endUserGeozone' or 'productSeries' in this context.

    Limitations:

    This particular join is currently limited to uni-dimensional index for both dataset involved in the join.
    The index being the joining key.


    usage :
    perform a join with df_left as data and df_right as a referential, considering default values in a referential.
    df_left and df_right are meant be indexed by their joining key (which should be uni-dimensional).
    using the option lindex and rindex, one can set the index on the fly.
    right_column is the  column containing the info to add to df_left(the data)
    audited allows audited_join to be called instead of usual join.
    default_value is the default value used in the referential.
    """
    if lindex:
        df_left = df_left.set_index(lindex)
    if rindex:
        df_right = df_right.set_index(rindex)

    df_right = df_right[[right_column]]

    lindex_name = df_left.index.name
    out = []

    df_left_with_key_index = df_left.index.intersection(df_right.index.drop(default_value))
    df_left_with_key = df_left.loc[df_left_with_key_index]
    df_left_with_default_index = df_left.index.difference(df_left_with_key_index)
    df_left_with_default = df_left.loc[df_left_with_default_index]
    # performing the 'normal join'
    keyed_join = df_left_with_key.join(df_right, how=how, lsuffix=lsuffix, rsuffix=rsuffix, sort=sort) if not audited \
        else audited_join(df_left_with_key, df_right, how=how, lsuffix=lsuffix, rsuffix=rsuffix, sort=sort)
    keyed_join.reset_index(inplace=True)
    if 'index' in keyed_join.columns:
        keyed_join.rename(columns={'index': lindex_name}, inplace=True)
    out.append(keyed_join)
    # adding the default values
    df_left_with_default[right_column] = pd.Series(df_right.loc[default_value, right_column],
                                                   index=df_left_with_default.index)
    out.append(df_left_with_default.reset_index())
    return pd.concat(out)


def format_column_name(column_name):
    name_parts = [X.replace('ID', 'id') for X in column_name.split('_')]
    first_part = name_parts[0][0].lower() + name_parts[0][1:]
    other_parts = [X[0].upper() + X[1:] for X in name_parts[1:]]
    return ''.join([first_part] + other_parts)


def dwh_to_di_renaming_rule(dataframe):
    return {X: format_column_name(X) for X in dataframe.columns}


def rename_columns(dataframe, input_format='DWH', output_format='DI'):
    print('conversion', input_format, 'to', output_format)
    return dataframe.rename(columns=dwh_to_di_renaming_rule(dataframe=dataframe))


load_info()
