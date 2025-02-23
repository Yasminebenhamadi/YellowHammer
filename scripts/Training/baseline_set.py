import sys, os
import librosa
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy.signal import hilbert, butter, lfilter
from sklearn.metrics import precision_recall_curve

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Preprocessing'))

from folds import *


# I got the code from here https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html 
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band') # from scipy.signal
    y = lfilter(b, a, data)
    return y

def db_rmse_energy(signal):
    origine = np.sqrt(signal@signal.T/signal.shape[0])
    db = librosa.amplitude_to_db(origine)
    return db

def values_db (folder_db,files, exclude_far=False):
    if exclude_far:
        files = [file for file in files if "100m" not in file]
        files = [file for file in files if "150m" not in file]
        files = [file for file in files if "200m" not in file]

    results = []

    for song_file in files:
        samplerate, data = wavfile.read(folder_db+song_file)
        data = data.astype(float)
        data = butter_bandpass_filter(data, 4000, 9000, samplerate, order=6)
        
        if data.shape[0] != 0.5*samplerate:
            print("Not a 500ms clip!")

        results.append(db_rmse_energy(data))
        
    return results

def get_baseline(folder_positive, folder_negative, df_folds, outfolder):
    neg_train, neg_val, neg_test = data_split(folder_negative)

    fold_thresholds = []

    for fld_nb in range(df_folds.shape[0]):
        files_train, files_validate, files_test = get_fold_split(folder_positive, df_folds[fld_nb])
        
        energy_true = values_db (folder_positive, files_train, exclude_far=False)
        energy_false = values_db (folder_negative, neg_train, exclude_far=False)

        # Get threshold based on quantiles    
        threshold_quantile = round((np.quantile(energy_true, 0.25) + np.quantile(energy_false, 0.75))/2, 2)


        # Get threshold based on precision_recall curve
        y_true = np.concatenate([[1 for e in energy_true], [0 for e in energy_false]])
        y_scores = np.concatenate([energy_true, energy_false])

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores) # Computes precision-recall pairs for different thresholds values
        f1_score = 2*precision*recall/(precision+recall+1e-10) #1e-10 ESP for div /0
        indx_max = np.argmax(f1_score)
        threshold_curve = round(thresholds[indx_max],2)

        # Saving thresholds
        fold_thresholds.append([threshold_quantile, threshold_curve])


    df_threshold = pd.DataFrame(np.array(fold_thresholds), columns=['threshold_quantile', 'threshold_curve'])
    df_threshold.to_csv(outfolder+"baseline_thresholds.csv", index=False)

if __name__ == "__main__":
    folder_positive = "/Users/yasminebenhamadi/YellowHammer/Data/positive/"
    folder_negative = "/Users/yasminebenhamadi/YellowHammer/Data/negative/"

    df_folds = pd.read_csv('/Users/yasminebenhamadi/YellowHammer/Processed/folds.csv')
    df_folds = df_folds.to_numpy()

    models_folder = "/Users/yasminebenhamadi/YellowHammer/models/"
    os.makedirs(models_folder, exist_ok = True)

    get_baseline(folder_positive, folder_negative, df_folds, models_folder)
