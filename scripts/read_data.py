import h5py
import numpy as np 
import pandas as pd
from sklearn.utils import shuffle

# Parsing information from filenames based on the File Name Format defined by Ilaria as well as the preprocessing done

def get_is_biotic_neg(file_name):
    # Check if negative samplefile is a negative from another species
    return "bioneg_" in file_name

def parse_indiv(file_name): 
    # Get indivual ID number
    sub_strings = np.array(file_name.split("_"))
    index = np.where(sub_strings == 'YH')[0][0]
    return int(sub_strings[index+1])

def parse_distance(file_name):
    # Get distance as a float
    distance_strings = ["_6_5m_", "_12_5m", "_25m_", "_50m_", "_100m_", "_150m_", "_200m_"]
    distance_values = np.array([6.5, 12.5, 25, 50, 100, 150, 200])

    check_distance = np.array([d in file_name for d in distance_strings])

    distance = distance_values[check_distance]
    if "YH" in file_name and distance.shape[0] != 0:
        return distance[0]
    elif get_is_biotic_neg(file_name):
        return -1
    else:
        return -2

def get_folds_data(folder_process, nb_fold, data_name, transpose=False, augment=True):
    '''
    To get training data from the .h5 files. Train, val, test are read are seperately because Negative samples are all stored together to save space. 
    data_name is either 'train', 'test' or 'val'  
    '''
    pos_file_name = folder_process+"fold_"+str(nb_fold)+".h5f"
    neg_file_name = folder_process+"negatives.h5f"

    # Read 'train', 'test' or 'val' data from fold .h5 file
    pos_data_file = h5py.File(pos_file_name, 'r')
    X_pos =np.array(pos_data_file[data_name])

    # Read 'train', 'test' or 'val' data from negatives .h5 file
    neg_data_file = h5py.File(neg_file_name, 'r')
    X_neg =np.array(neg_data_file[data_name])

    X = np.concatenate([X_pos, X_neg])
    
    if transpose:
        X = X.transpose(0, 2, 1)

    y = np.concatenate([[1 for i in range(X_pos.shape[0])], [0 for i in range(X_neg.shape[0])]])

    name_suffix = "_"+data_name+"_names.csv"
    df_names_pos = pd.read_csv(folder_process+"fold_"+str(nb_fold)+name_suffix).to_numpy().flatten()
    df_names_neg = pd.read_csv(folder_process+"negatives"+name_suffix).to_numpy().flatten()
    file_names = np.concatenate([list(df_names_pos), list(df_names_neg)])

    if not augment:
        indx_og = [i for i in range(len(file_names)) if not "cascade" in file_names[i] and not "faint" in file_names[i] and not "cstart" in file_names[i]]
        X, y, file_names = X[indx_og], y[indx_og], file_names[indx_og]

    return shuffle(X, y, file_names, random_state=nb_fold)