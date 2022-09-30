import os

from fedot_wrapper import FedotWrapper

BASE_DATA_PATH = os.path.join(os.getcwd(), 'data')

if __name__ == '__main__':
    FedotWrapper(path_to_data_dir=BASE_DATA_PATH).run(fit_type='normal')
