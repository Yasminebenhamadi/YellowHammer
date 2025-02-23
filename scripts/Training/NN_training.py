#!/usr/bin/env python
# coding: utf-8

import sys, os
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import models, layers, optimizers, losses, metrics

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from read_data import *

def create_bands_model(input_shape): # from students code
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Conv1D(32, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(64, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_avg_model(input_shape):
    model = models.Sequential([
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryAccuracy(),
            metrics.FalseNegatives(),
        ],
    )
    return model

def create_envelope_model(input_shape):
    model = models.Sequential([
        layers.InputLayer(shape=input_shape),
        layers.Conv1D(8, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(16, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryAccuracy(),
            metrics.FalseNegatives(),
        ],
    )
    return model

def create_mel_model(input_shape):
    model = models.Sequential([
        layers.InputLayer(shape=input_shape),
        layers.Conv1D(5, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Conv1D(10, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss=losses.BinaryCrossentropy(),
        metrics=[
            metrics.BinaryAccuracy(),
            metrics.FalseNegatives(),
        ],
    )
    return model

def folds_training(data_folder, create_model, outfolder, outname, transpose=False, nb_folds = 5, save=True, augment=True, expand=False):
    models = []
    for fl_nb in range(nb_folds):
        # Read data
        X_train, y_train, train_names = get_folds_data(data_folder, nb_fold=fl_nb, data_name='train', transpose=transpose, augment=augment)
        X_val, y_val, val_names = get_folds_data(data_folder, nb_fold=fl_nb, data_name='val', transpose=transpose, augment=augment)

        input_shape = X_train[0].shape
        if expand:
            input_shape = (input_shape[0], 1)

        model = create_model(input_shape)
        
        class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(y_train),y=y_train)
        class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

        history = model.fit(x=X_train,y=y_train,batch_size=32,epochs=30, validation_data=(X_val, y_val), shuffle=True, class_weight=class_weights_dict)
        
        models.append(model)
        if save:
            os.makedirs(outfolder, exist_ok = True)
            model.export(outfolder+outname+"_"+str(fl_nb))
            model.save(outfolder+outname+"_"+str(fl_nb)+"/model.h5")
    return models

def train_run(process_name, create_model, transpose=False, augment=True, expand=False):
    process_folder_all = "/Users/yasminebenhamadi/YellowHammer/Processed/"
    outfolder_all = "/Users/yasminebenhamadi/YellowHammer/models/"

    process_folder = process_folder_all+process_name+"/"
    outfolder = outfolder_all+process_name+"_models/"
    models = folds_training(process_folder, create_model, outfolder=outfolder, outname=process_name+"_model", transpose=transpose, nb_folds = 1, save=True, augment=augment, expand=expand)
    print(models[0].summary())


if __name__ == "__main__":

    #train_run(process_name="average", create_model=create_avg_model)
    #train_run(process_name="envelope", create_model=create_envelope_model, expand=True)
    #train_run(process_name="bands", create_model=create_bands_model, expand=True)
    #train_run(process_name="mels", create_model=create_mel_model, transpose=True)
    