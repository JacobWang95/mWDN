import numpy as np 
import os

def get_cls_dict(cls):
    cls_list = sorted(np.unique(cls))
    cls_range = len(cls_list)
    cls_dict = {}
    for i in range(cls_range):
        cls_dict[cls_list[i]] = i
    return cls_dict

def clean_cls(cls, cls_dict):
    """ 
    For some particular datasets, the labels are not continuous
    Like 0, 1, 3, 4, 5
    Using this function to clean the labels
    """
    cls_list = []
    for c in cls:
        cls_list.append(cls_dict[c])
    return np.array(cls_list, int)

def read_data(root_dir, data_name, normalization=True):
    """ 
    Function for reading UCR time series dataset
    the root_dir should be like '/path/to/UCR_TS_Archive_2015'
    """

    data_dir = os.path.join(root_dir, data_name)

    data_train = np.loadtxt(os.path.join(data_dir, data_name+'_TRAIN'), delimiter=',')
    data_test = np.loadtxt(os.path.join(data_dir, data_name+'_TEST'), delimiter=',')

    cls_dict = get_cls_dict(data_train[:, 0])

    label_train = clean_cls(data_train[:, 0], cls_dict)
    label_test = clean_cls(data_test[:, 0], cls_dict)
    if normalization:
        mean_v = data_train[:, 1:].mean()
        std_v = data_train[:, 1:].std()

        input_train = np.array((data_train[:, 1:]-mean_v)/std_v, np.float32)
        input_test = np.array((data_test[:, 1:]-mean_v)/std_v, np.float32)
    else:
        input_train = np.array(data_train[:, 1:], np.float32)
        input_test = np.array(data_test[:, 1:], np.float32)

    return input_train, label_train, input_test, label_test

def padding_zeros(data_in, num_pad):
    a = np.zeros((len(data_in), num_pad))
    b = np.append(data_in, a, 1)
    return b


def load_data(DATA_ROOT, name_data):
    x_train, y_train, x_test, y_test = read_data(DATA_ROOT, name_data)

    num_padding = 16 - (len(x_test[0]) % 16)
    if num_padding != 0:
        x_train = padding_zeros(x_train, num_padding)
        x_test = padding_zeros(x_test, num_padding)
    return x_train, y_train, x_test, y_test


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx