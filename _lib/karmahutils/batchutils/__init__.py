from os import listdir, remove
import pandas as pd
import karmahutils as kut

version_info = "v0.1"
version_type = 'moonshade library'
authors = ['Yann Girard']
contact = 'yann.girard@gmail.com'
lib_name = 'karmahutils'
purpose = """The goal of this lib is to facilitate batching actions over a dataframe"""


def clean_backups(backup_dir, task_name):
    if backup_dir is None:
        return
    files = listdir(backup_dir)
    candidate_files = [X for X in files if task_name in X]
    [remove(backup_dir + X) for X in candidate_files]
    print('removed', len(candidate_files), 'files')
    return 1


def load_backup(backup_dir, task_name, with_cleaning=False, reset_job=False):
    if backup_dir is None:
        return []
    files = listdir(backup_dir)
    candidate_files = [X for X in files if task_name in X]

    if reset_job:
        clean_backups(backup_dir=backup_dir, task_name=task_name)
        return []

    if len(candidate_files) == 0:
        print('found no backups for task', task_name)
        return []

    selected = max(candidate_files)
    print('selected', selected)
    loaded_df = pd.read_csv(backup_dir + selected)
    return [loaded_df]


def batch_me(
        data,
        instructions,
        batch_size=100,
        track_offset=None,
        with_save_dir=None,
        with_load_dir=None,
        task_name='batchedJob',
        reset_job=False,
        clean_backup=True,
):
    from math import ceil, floor

    current_batch = 0

    kut.display_message('batching', instructions)

    if with_load_dir is True:
        with_load_dir = with_save_dir
    if with_save_dir:
        save_radical = with_save_dir + '_'.join([task_name, kut.file_stamp()])
        print('batch will be saved in:', save_radical + '_*')

    if type(instructions) is not list:
        instructions = [instructions]

    # I wanted to make batching instruction intuitive but you can't access a variable name inside a function
    # There might be a more convoluted way to do it , either through a class attribute or a wrapper. For now...
    # instructions=[X.replace(data,'batch_df') for X in instructions]

    remaining_data = data
    treated_stack = load_backup(backup_dir=with_load_dir, task_name=task_name, reset_job=reset_job)
    if len(treated_stack):
        treated_df = pd.concat(treated_stack, ignore_index=True)
        print('found', len(treated_df), 'backed_up rows')
        remaining_data = data.loc[data.index.difference(treated_df.index)]
        print('remaining to treat', len(remaining_data), 'rows')

    if len(remaining_data):
        batch_size = len(remaining_data) if batch_size > len(remaining_data) else batch_size
        total_batches = ceil(len(remaining_data) / batch_size)

        print(floor(len(remaining_data) / batch_size), 'batches of', batch_size, 'rows')
        if track_offset is None:
            track_offset = max(round(0.25 * total_batches), 1)
        if len(remaining_data) % batch_size:
            print('one batch of', len(remaining_data) % batch_size, 'rows')
        if track_offset:
            print('track offset every', track_offset, 'batch')

    else:
        print('nothing left to batch')

    for batch_df in kut.split_me(data=remaining_data, batch_size=batch_size):
        current_batch += 1

        for instruction in instructions:
            eval(instruction)

        treated_stack.append(batch_df)

        # intermediate prints and saves
        print(current_batch, track_offset)
        if not current_batch % track_offset:
            print('done batch', current_batch)
            if with_save_dir is not None:
                save_name = '_'.join([save_radical, str(current_batch)])
                current_treated_df = pd.concat(treated_stack, ignore_index=True)
                current_treated_df.to_csv(save_name, index=False)
                print('saved', len(current_treated_df), 'rows')
                print(save_name)

    out = pd.concat(treated_stack, ignore_index=True)
    if clean_backup:
        kut.display_message('cleaning backups')
        clean_backups(backup_dir=with_load_dir, task_name=task_name)
    return out
