#!/usr/bin/env python
# coding: utf-8

import os
import h5py
import numpy as np 
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

# Code needed for splitting data and performing/saving preprocessing results

def get_files(folder, id_list):
    # Get all the files containing the vocalizations from individuals in id_list
    files = os.listdir(folder)
    list_files = []
    for id_indiv in id_list:
        yes_files = [f for f in files if "YH_"+str(id_indiv)+"_" in f]
        list_files.extend(yes_files)
    return list_files

# Train, validation, test split
def folds_spilt(outfolder):
    # Generate folds based on individuals
    X = np.array(range(1,11)) # the 10 individuals
    y = np.ones(X.shape)

    skf1 = StratifiedKFold(n_splits=5, shuffle=True, random_state=10) # Train, test split

    folds = []

    for i, (train_index, test_index) in enumerate(skf1.split(X, y)):
        test = X[test_index]

        # Train, val split
        shift_add = 5 #random.randint(1,9)
        if test_index[1] == (test_index[0] + shift_add) % 10 or test_index[0] == (test_index[1] + shift_add) % 10:
            shift_add = shift_add + 1 # To make sure an indiv is not in test and val as the same time

        validate_index = (np.array(test_index) + shift_add) % 10
        validate = X[validate_index]

        train_index = [i for i in train_index if i not in validate_index]
        train = X[train_index]
        
        # Checking that intersect(val, test) is empty
        check = [x for x in X if x in validate and x in test]
        assert(len(check)==0)

        folds.append(np.concatenate([train, test, validate]))

    # Save fold into CSV
    folds = np.array(folds)
    colums = ["tr"+str(i) for i in range(1,7)]
    colums.extend(["v"+str(i) for i in range(1,3)])
    colums.extend(["t"+str(i) for i in range(1,3)])

    pd.DataFrame(data=folds, columns=colums).to_csv(outfolder+'folds.csv', index=False)

def get_fold_split(folder, fold):
    # Get audio files split from the individual wise stratified k-fold split expressed in a list of [6 train IDs, 2 val IDs, 2 test IDs] (totaling the 10 individuals)    
    fold_train, fold_valid, fold_test = fold[0:6], fold[6:8], fold[8:10]

    files_train = get_files(folder, fold_train)
    files_validate = get_files(folder, fold_valid)
    files_test = get_files(folder, fold_test)

    return files_train, files_validate, files_test

def data_split(folder, labels=None): 
    # Get (0.8, 0.1, 0.1) train, validation, test audio files split
    files = os.listdir(folder)
    files = [f for f in files if ".wav"]

    if labels == None:
        y = [0 for f in files] # for when I'm only splitting the negatives
    else:
        y = labels  # in case we want to do the split on the entire dataset

    X_train, X_test, y_train, y_test = train_test_split(files, y, test_size=0.2, random_state=1)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

    return np.array(X_train), np.array(X_val), np.array(X_test)

# Process and save
def save_h5_data(process_method, folder, train_files, val_files, test_files, outfile):
    '''
    Method for saving the processed data. For both negative and positive samples data is saved to a .h5 file and names are saved into separate csv files (separately). 

    process_method: the function to process that data with. 
    outfile: should indicate the path as well as the first part of the file name

    '''
    X_train = process_method(folder, train_files)
    X_val = process_method(folder, val_files)
    X_test = process_method(folder, test_files)

    # Saving split as .h5 file
    h5f = h5py.File(outfile+".h5f", 'w')
    h5f.create_dataset('train', data=X_train)
    h5f.create_dataset('val', data=X_val)
    h5f.create_dataset('test', data=X_test)
    h5f.close()

    # Saving names as csv files
    names_train = pd.DataFrame(np.array(train_files))
    names_train.to_csv(outfile+"_train_names.csv", index=False)

    names_val = pd.DataFrame(np.array(val_files))
    names_val.to_csv(outfile+"_val_names.csv", index=False)

    names_test = pd.DataFrame(np.array(test_files))
    names_test.to_csv(outfile+"_test_names.csv", index=False)

# Process and save individual wise stratified k-fold data
def process_folds_data(process_method, folder_positive, folder_negative, folder_process, df_folds):
    os.makedirs(folder_process, exist_ok = True)
    
    # save negatives file (same across folds)
    neg_train_files, neg_val_files, neg_test_files = data_split(folder_negative)
    outfile_neg = folder_process+"negatives"
    save_h5_data(process_method, folder_negative, neg_train_files, neg_val_files, neg_test_files, outfile_neg)

    # save each fold in a seperate file
    for fld_nb in range(df_folds.shape[0]):
        pos_train_files, pos_val_files, pos_test_files = get_fold_split(folder_positive, df_folds[fld_nb])

        outfile_pos = folder_process+"fold_"+str(fld_nb)
        save_h5_data(process_method, folder_positive, pos_train_files, pos_val_files, pos_test_files, outfile_pos)