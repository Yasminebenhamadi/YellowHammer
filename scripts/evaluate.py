import sys,os
import h5py
import tracemalloc
import numpy as np 
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tensorflow.keras import models, layers, optimizers, losses, metrics
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report, precision_recall_curve

sys.path.append(os.path.join(os.path.dirname(__file__), 'Preprocessing'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Training'))

from read_data import *
from folds import *
from baseline_set import *

# Evaluation methods
def recall_at_ditance(y_test, y_pred, d_test):
    df_results = pd.DataFrame(np.array([y_test, y_pred, d_test]).T, columns=['true','predict', 'distance'])

    recalls = []
    for distance, frame in df_results.groupby(['distance']):
        if distance[0] > 0:
            true_restrict = frame['true']
            predict_restrict = frame['predict']
            recalls.append(recall_score(true_restrict, predict_restrict, average="binary"))
    return recalls

def precision_mid_recall_at_ditance(y_test, y_pred_scores, d_test):
    df_results = pd.DataFrame(np.array([y_test, y_pred_scores, d_test]).T, columns=['true','predict', 'distance'])

    medians = []
    precisions = []
    for distance, frame in df_results.groupby(['distance']):
        if distance[0] > 0:
            true_restrict = frame['true']
            predict_restrict = frame['predict']
            median_value = np.quantile(predict_restrict, 0.5)
            if median_value ==1:
                median_value = 0.999999

            y_pred = (y_pred_scores>=median_value).astype(int).flatten()
            precisions.append(precision_score(y_test, y_pred, average="binary"))
            medians.append(median_value)
    return precisions

# Plot functions
def plot_score_hist(eval_plot_folder, score_list, distance_list, model_name):
    df_results = pd.DataFrame(np.array([score_list, distance_list]).T, columns=['score', 'distance'])
    
    negatives_scores = []

    fig, axs = plt.subplots(1,7, figsize=(25, 3))

    i = 0
    for distance, frame in df_results.groupby(['distance']):
        if distance[0] < 0:
            negatives_scores = frame['score']
        else:
            distance_scores = frame['score']
            axs[i].hist(distance_scores, density=True, label="Positive", alpha=0.5)
            axs[i].hist(negatives_scores, density=True, label="Negative", alpha=0.5)
            axs[i].set_title("Scores at "+str(distance[0]), y=-0.2)
            
            i = i + 1
    
    plt.legend()
    plt.suptitle('Histogram of (predicted) scores of '+ model_name)
    folder_save = eval_plot_folder+"histograms/"
    os.makedirs(folder_save, exist_ok=True)
    plt.savefig(folder_save+model_name+'.png')
    plt.clf()

def plot_conf_matrix(eval_plot_folder, model_name, y_test=None, y_pred=None, conf_matrix=None):
    plt.figure(figsize=(6.4, 4.8))
    if conf_matrix == None:
        conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix "+model_name)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    folder_save = eval_plot_folder+"conf_matrices/"
    os.makedirs(folder_save, exist_ok=True)
    plt.savefig(folder_save+model_name+'.png')
    plt.clf()

def plot_curve_at_distance(eval_plot_folder, list_all, names, plot_name):
    plt.figure(figsize=(6.4, 4.8))
    for r in range(len(list_all)):
        plt.plot(list_all[r], '-o', label=names[r], linestyle="dotted")
    distances_unique = np.array([6.5, 12.5, 25, 50, 100, 150, 200])
    plt.legend()
    plt.xticks(range(distances_unique.shape[0]),distances_unique)
    plt.xlabel("Distance (m)")
    plt.ylabel(plot_name)    
    plt.savefig(eval_plot_folder+plot_name+'.png')
    plt.clf()

def plot_ordered(eval_plot_folder, list_plot, names, title, color, ylabel="", show=True):
    list_plot = np.array(list_plot).flatten()
    names = np.array(names)
    sort_indx = list_plot.argsort()

    plt.figure(figsize=(15, 10), dpi=80)

    list_plot_sorted = np.round(list_plot[sort_indx], 2)

    plt.bar(range(len(list_plot)), list_plot_sorted, width=0.2, color=color)
    for i in range(len(list_plot)):
        plt.text(i,list_plot_sorted[i]*1.01,list_plot_sorted[i], ha = 'center')
    plt.xticks(range(len(list_plot)), names[sort_indx])
    plt.title(title)
    plt.xlabel("Models")
    plt.ylabel(ylabel)
    plt.savefig(eval_plot_folder+title+'.png')
    plt.clf()

# Inference and evaluation
''' Baseline'''
def get_baseline(models_folder, fold, eval_plot_folder_fld, plot_hist=True, plot_conf=False, plot=False):
    folder_positive = "/Users/yasminebenhamadi/YellowHammer/Data/positive/"
    folder_negative = "/Users/yasminebenhamadi/YellowHammer/Data/negative/"

    df_folds = pd.read_csv('/Users/yasminebenhamadi/YellowHammer/Processed/folds.csv')
    df_folds = df_folds.to_numpy()

    df_threshold = pd.read_csv(models_folder+"baseline_thresholds.csv")

    _, _, neg_test = data_split(folder_negative)

    df_t = df_threshold.iloc[fold]
    threshold_quantile, threshold_curve = df_t['threshold_quantile'], df_t['threshold_curve']
    
    _, _, files_test = get_fold_split(folder_positive, df_folds[fold])

    files_test = [file for file in files_test if not "cascade" in file and not "faint" in file and not "cstart" in file]

    # Baseline predictions
    energy_true = values_db (folder_positive,files_test, exclude_far=False)
    energy_false = values_db (folder_negative,neg_test, exclude_far=False)
    y_scores = np.concatenate([energy_true, energy_false])

    y_true = np.concatenate([[1 for e in energy_true], [0 for e in energy_false]])
    y_pred_quantile = y_scores > threshold_quantile
    y_pred_curve = y_scores > threshold_curve
    distance = np.concatenate([[parse_distance(f) for f in files_test], [-1 for e in energy_false]])

    if plot:
        plot_baseline(folder_positive,files_test, folder_negative,neg_test, threshold_quantile, threshold_curve, fold)

    recall_quantile = recall_at_ditance(y_true, y_pred_quantile, distance)
    recall_curve = recall_at_ditance(y_true, y_pred_curve, distance)

    precision_mid = precision_mid_recall_at_ditance(y_true, y_scores, distance)

    precision_quantile = precision_score(y_true, y_pred_quantile, average="binary")
    precision_curve = precision_score(y_true, y_pred_curve, average="binary")

    f1_score_quantile = f1_score(y_true, y_pred_quantile, average="binary")
    f1_score_curve = f1_score(y_true, y_pred_curve, average="binary")

    accuracy_score_quantile = accuracy_score(y_true, y_pred_quantile)
    accuracy_score_curve = accuracy_score(y_true, y_pred_curve)


    if plot_hist:
        plot_score_hist(eval_plot_folder_fld, y_scores, distance, 'Baseline (RMSE energy)')
    if plot_conf:
        plot_conf_matrix(eval_plot_folder_fld, "Baseline quantile", y_test=y_true, y_pred=y_pred_quantile)
        plot_conf_matrix(eval_plot_folder_fld, "Baseline curve", y_test=y_true, y_pred=y_pred_curve)

    return recall_quantile, recall_curve, precision_mid, precision_quantile, precision_curve, f1_score_quantile, f1_score_curve, accuracy_score_quantile, accuracy_score_curve


''' TF models'''
def get_NN_results (data_folder, fold, model_path, transpose):
    X_test, y_test, names = get_folds_data(data_folder, nb_fold=fold, data_name='test', transpose=transpose, augment=False)
    model = tf.keras.models.load_model(model_path)
    y_pred_scores = model.predict(X_test).flatten()
    y_pred = (y_pred_scores>0.5).astype(int).flatten()
    return y_test, y_pred, y_pred_scores, names

# Evaluation for TF models
def eval_results(data_folder, fold, model_path, transpose, model_name, eval_plot_folder_fld, plot_hist=True, plot_conf=False):
    y_test, y_pred, y_pred_scores, names = get_NN_results (data_folder, fold, model_path, transpose)
    d_test = [parse_distance(file_name) for file_name in names]
    recalls_at = recall_at_ditance(y_test, y_pred, d_test)
    precisions_at = precision_mid_recall_at_ditance(y_test, y_pred_scores, d_test)
    precision = precision_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    accuracy = accuracy_score(y_test, y_pred)
    if plot_hist:
        plot_score_hist(eval_plot_folder_fld, y_pred_scores, d_test, model_name)
    if plot_conf:
        plot_conf_matrix(eval_plot_folder_fld, model_name, y_test=y_test, y_pred=y_pred)
    
    return recalls_at, precisions_at, precision, f1, accuracy



def compare_methods():
    data_folder_all = "/Users/yasminebenhamadi/YellowHammer/Processed/"
    models_folder_all = "/Users/yasminebenhamadi/YellowHammer/models/"
    eval_plot_folder = "/Users/yasminebenhamadi/YellowHammer/plots/"

    process_names = ["average", "envelope", "bands", "mels"]

    data_folders = [data_folder_all+process_name+"/" for process_name in process_names]
    models_folders = [models_folder_all+process_name+"_models/" for process_name in process_names]

    for fld_nb in range(1):
        eval_plot_folder_fld=eval_plot_folder+"fold_"+str(fld_nb)+"/"
        os.makedirs(eval_plot_folder_fld, exist_ok=True)
        fld_models_folders = [models_folders[i]+process_names[i]+"_model_"+str(fld_nb)+"/model.h5" for i in range(len(process_names))]
        # Read data
        precision_list = []
        f1_list = []
        accuracy_list = []

        all_recalls = []
        all_precisions = []

        hist = True
        conf = True

        for i in range(len(process_names)):
            recalls_at, precisions_at, precision, f1, accuracy = eval_results(data_folders[i], fld_nb, fld_models_folders[i], transpose=(i==3), model_name=process_names[i], eval_plot_folder_fld=eval_plot_folder_fld, plot_hist=hist, plot_conf=conf)
            all_recalls.append(recalls_at)
            all_precisions.append(precisions_at)
            precision_list.append(precision)
            f1_list.append(f1)
            accuracy_list.append(accuracy)

        add_baseline = True
        if add_baseline:
            recall_quantile, recall_curve, precision_mid, precision_quantile, precision_curve, f1_score_quantile, f1_score_curve, accuracy_score_quantile, accuracy_score_curve = get_baseline(models_folder_all,fld_nb,eval_plot_folder_fld=eval_plot_folder_fld,plot_hist=hist, plot_conf=conf)
            all_recalls.extend([recall_quantile, recall_curve])
            all_precisions.extend([precision_mid, precision_mid])

            precision_list.extend([precision_quantile, precision_curve])
            f1_list.extend([f1_score_quantile, f1_score_curve])
            accuracy_list.extend([accuracy_score_quantile, accuracy_score_curve])
            process_names.extend(['base_quantile', 'base_curve'])



        plot_curve_at_distance(eval_plot_folder_fld, all_recalls, process_names, "Recall at distance")
        plot_curve_at_distance(eval_plot_folder_fld, all_precisions, process_names, "Precision at distance")

        plot_ordered(eval_plot_folder_fld, precision_list, process_names, "Precision", color="green")
        plot_ordered(eval_plot_folder_fld, f1_list, process_names, "F1 score", color="red")
        plot_ordered(eval_plot_folder_fld, accuracy_list, process_names, "Accuracy", color="blue")


if __name__ == "__main__":
    compare_methods()