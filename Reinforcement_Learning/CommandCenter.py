import pandas as pd
import numpy as np
import os
import time
import shutil

class CommandCenter:

    def __init__(self, base, model_base, data_base, worker=False, trainer=False, reset_worker=False, reset_trainer=False):
        self.global_base_path = base
        self.global_model_path = model_base
        self.global_data_path = data_base

        self.sign_in_sheet_name = 'WorkerSignInSheet.csv'
        self.trainer_sheet_name = 'TrainerSheet.npy'

        # self.model_version_start = 1
        self.model_version = 1
        self.model_increment = 0.01

        if reset_worker:
            self._reset_worker_sheet()

        if reset_trainer:
            self._create_trainer_sheet()

        self.worker = worker
        self.trainer = trainer

        self._load_trainer_sheet()
        self._get_model_version()

        if self.worker:
            self._open_signin_sheet()
            self._find_next_worker()
            self._update_signin_sheet()
            self._save_signin_sheet()
            self.update_worker_paths()

        if self.trainer:
           self.update_trainer_paths()
           self.trainer_find_data_filepaths()
           self.model_update_time = 1  # Update time in Minutes
           self.last_update = time.time()

    def check_if_increment_trainer(self):
        if self._model_time_update():
            self._increment_model()
            self.update_trainer_paths()
            self._update_trainer_sheet()
            self._save_trainer_sheet()
            self.trainer_find_data_filepaths()
            print('Updated model to version {}'.format(self.model_version))
            return True

    def update_worker_paths(self):
        self._load_trainer_sheet()
        self._get_model_version()
        self.data_path = self.global_data_path + '/' + 'worker_' + str(self.worker_id) + '/rev_' + str(
            self.model_version) + '/'
        self._get_model_path()

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

    def update_trainer_paths(self):
        self._get_model_path()

    def trainer_find_data_filepaths(self):
        self.all_valid_data_paths = []
        files = os.listdir(self.global_data_path)

        worker_idxs = find_string_idx(files, 'worker')
        worker_files = np.array(files)[worker_idxs]

        prev_models_to_train = 3
        start_rev = self.model_version - (prev_models_to_train * self.model_increment)

        revs_to_train = np.arange(start_rev, self.model_version, step=self.model_increment)
        revs_to_train = np.round(revs_to_train, 2)

        for rev in revs_to_train:
            rev = float(rev)

            for worker_file in worker_files:
                worker_path = self.global_data_path + '/' + worker_file + '/'

                try:
                    worker_path = worker_path + 'rev_' + str(rev) + '/'
                    all_files_in_worker = os.listdir(worker_path)
                    data_files_idx = find_string_idx(all_files_in_worker, 'data_')
                    data_files = np.array(all_files_in_worker)[data_files_idx]

                    for file in data_files:
                        valid_file = worker_path + file + '/'
                        self.all_valid_data_paths.append(valid_file)

                except:
                    pass

    def _reset_worker_sheet(self):
        self._open_signin_sheet()
        self.sign_in_sheet[:,1] = 0
        self._save_signin_sheet()

    def _open_signin_sheet(self, sheetname=None):
        if sheetname is not None:
            self.sign_in_sheet_name = sheetname

        self.sign_in_sheet_path = self.global_base_path + '/' + self.sign_in_sheet_name
        self.sign_in_sheet = np.array(pd.read_csv(self.sign_in_sheet_path, header=None)).astype(np.int8)

    def _find_next_worker(self):
        free_idx = np.where(self.sign_in_sheet[:, 1] == 0)[0]
        idx = free_idx[0]
        self.worker_id = self.sign_in_sheet[idx][0]

    def _update_signin_sheet(self):
        idx = np.where(self.sign_in_sheet[:, 0] == self.worker_id)[0][0]
        self.sign_in_sheet[idx][1] = 1

    def _save_signin_sheet(self):
        np.savetxt(self.sign_in_sheet_path, self.sign_in_sheet, delimiter=',')

    def _create_trainer_sheet(self):
        print('Creating New Trainer Sheet at {}'.format(self.global_data_path + '/' + self.trainer_sheet_name))
        self._update_trainer_sheet()
        self._save_trainer_sheet()

    def _update_trainer_sheet(self):
        self.trainer_controls = {'Model Version': self.model_version}

    def _save_trainer_sheet(self):
        print('Saved Trainer Sheet')
        print('Saved to: {}'.format(self.global_data_path + '/' + self.trainer_sheet_name))

        np.save(self.global_data_path + '/' + self.trainer_sheet_name, self.trainer_controls)

    def _load_trainer_sheet(self):
        loaded = False
        while loaded is False:
            try:
                self.trainer_controls = np.load(self.global_data_path + '/' + self.trainer_sheet_name).item()
                loaded = True
            except OSError:
                time.sleep(1)

    def _get_model_version(self):
        self.model_version = self.trainer_controls['Model Version']
        self.next_model_version = self.model_version + self.model_increment

        # Convert to float so that versions are converted to strings properly
        self.model_version = np.round(float(self.model_version), 2)
        self.next_model_version = np.round(float(self.next_model_version), 2)

    def _get_model_path(self):
        self.model_path = self.global_model_path + '/rev_' + str(self.model_version) + '/'
        self.next_model_path = self.global_model_path + '/rev_' + str(self.next_model_version) + '/'

    def _model_time_update(self):
        update = (time.time() - self.last_update) > (self.model_update_time * 60)
        if update:
            self.last_update = time.time()
        return update

    def _increment_model(self):
        self.model_version += self.model_increment
        self.next_model_version += self.model_increment

        # Convert to float so that versions are converted to strings properly
        self.model_version = np.round(float(self.model_version), 2)
        self.next_model_version = np.round(float(self.next_model_version), 2)

def find_string_idx(list_strings, find_string):
    return [list_strings.index(i) for i in list_strings if find_string in i]