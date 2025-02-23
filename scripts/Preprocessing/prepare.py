import sys, os
import librosa
import textgrid
import numpy as np
import soundfile as sf
from functools import *
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from read_data import *
from preprocessing import * 
from folds import * 

''' 
    Data augmentation methods
''' 

# These data augmentation will be performed individually

def faint_start(sequence, samplerate, shift_seconds=0.2, scale=0.1, noise_level=0.002):
    # Make the first shift_seconds faint by scale factor and add noise
    end = int(shift_seconds*samplerate)
    scale = [scale if i<end else 1 for i in range(len(sequence))]
    additive_noise = [noise_level*np.random.normal() if i<end else 0 for i in range(len(sequence))]
    return sequence*scale + additive_noise

def clip_start(sequence, samplerate, shift_seconds=0.3, noise_level=0.002):
    # Clips the first shift_seconds while adding noise at the beginning
    end = int(shift_seconds*samplerate)
    scale = [1 if i<end else 0 for i in range(len(sequence))]
    additive_noise = [0 if i<end else noise_level*np.random.normal() for i in range(len(sequence))]
    return np.roll(sequence*scale+additive_noise, -end, axis=0)


# These data augmentation will be performed all together (code written by Master students)
def add_noise(sequence, noise_level=0.02):
    # Adds Gaussian noise to the envelope sequence.
    return sequence + noise_level * np.random.randn(*sequence.shape)

def time_shift(sequence, samplerate, max_shift_seconds=0.2):
    #Circularly shifts the audio envelope left or right by up to max_shift_seconds
    shift_max = int(samplerate*max_shift_seconds)
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(sequence, shift, axis=0)

def random_scaling(sequence, scale_range=(0.8, 1.2)):
    #Randomly scales the amplitude of the envelope.
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return sequence * scale

def augment_cascade(audio, samplerate, with_noise=True): 
    augmented_audio = random_scaling(audio)
    augmented_audio = time_shift(augmented_audio, samplerate) # doing time_shift before time_stretch because I'm defining shift_max in seconds
    if with_noise: 
        augmented_audio = add_noise(augmented_audio)
    return augmented_audio

'''Clipping and augmenting audio'''


def audio_clips(folder, outfolder, suffix="", maxclips=-1, dir_exist_ok=False, augment=False, clipsize=0.5):
    # Divide recordings into clipsize(seconds) clips

    os.makedirs(outfolder, exist_ok=dir_exist_ok)
    
    files = os.listdir(folder)
    files = [f for f in files if ".wav" in f]
    files = [f for f in files if "speaker" not in f]

    for sound_file in files:
        data, samplerate = librosa.load(folder+sound_file, sr=None)

        if data.shape[0]/samplerate >= clipsize:
            nb_clips = int(data.shape[0]/(samplerate*clipsize))
            if maxclips > 0:
                nb_clips = np.min([nb_clips,maxclips])

            start = 0
            end = int(clipsize*samplerate)
            for clip in range(nb_clips):
                data_clipped = data[start:end]
                clip_filename=suffix+"clip_"+str(clip+1)+"_"+sound_file

                sf.write(outfolder+clip_filename, data_clipped, samplerate)
                if augment:
                    with_noise = parse_distance(sound_file)<100 # Only adding noise to close distance clips
                    cascade = augment_cascade(data_clipped, samplerate, with_noise=with_noise)
                    sf.write(outfolder+"cascade_"+clip_filename, cascade, samplerate)
                
                    cstart = clip_start(data_clipped, samplerate)
                    sf.write(outfolder+"cstart_"+clip_filename, cstart, samplerate)

                    if clip > 0: # Dimming and clipping the start for 2nd and 3rd clips only
                        faint = faint_start(data_clipped, samplerate)
                        sf.write(outfolder+"faint_"+clip_filename, faint, samplerate)

                start=end
                end = end + int(clipsize*samplerate)

def prepare_negatives(grid_folder, negative_folder):
    # Extract negatives from the full recordings
    if not os.path.exists(negative_folder):
        os.makedirs(negative_folder)

        files = os.listdir(grid_folder)
        files = [f for f in files if ".wav" in f]
        files = [f for f in files if "speaker" not in f]
        
        for sound_file in files:
            data, samplerate = librosa.load(grid_folder+sound_file, sr=None)
            
            grid_file = grid_folder + sound_file[:-4]+".TextGrid"
            tg = textgrid.TextGrid.fromFile(grid_file)

            k = 0
            for i in range(len(tg.tiers)):
                for j in range(len(tg.tiers[i])):
                    if "YH" not in tg.tiers[i][j].mark:
                        negative = 'none'
                        if len(tg.tiers[i][j].mark.strip())!=0:
                            negative = tg.tiers[i][j].mark
                        xmin = tg.tiers[i][j].minTime
                        xmax = tg.tiers[i][j].maxTime
                        
                        data_negative = data[int(xmin*samplerate):int(xmax*samplerate)]
                        if data_negative.shape[0]/samplerate >= 0.5:
                            nb_clips = int(data_negative.shape[0]/(samplerate*0.5))

                            start = 0
                            end = int(0.5*samplerate)
                            for clip in range(nb_clips):
                                data_clipped = data_negative[start:end]
                                sf.write(negative_folder+"clip_"+str(clip+1)+"_"+negative+"_"+str(k)+"_"+sound_file, data_clipped, samplerate)
                                start=end
                                end = end + int(0.5*samplerate)
                            k=k+1

    else:
        print("Folder: "+negative_folder+" exists already.")

def prepare(positive_folder, negative_folder):
    # Divide recordings into 500ms clips 
    folder = "/Users/yasminebenhamadi/YellowHammer/Training_set_YH_songs/ALL_Songs_CUT/"

    # Positives
    if not os.path.isdir(positive_folder):
        audio_clips(folder, positive_folder, suffix="", clipsize=0.5, maxclips=3, augment=True, dir_exist_ok=True)

    if not os.path.isdir(negative_folder):
        # Abiotic negatives
        grid_folder = "/Users/yasminebenhamadi/YellowHammer/Training_set_YH_songs/recordings_textgrid_on_axis/"
        prepare_negatives(grid_folder, negative_folder)
        
        # Biotic negatives
        bioneg_folder = "/Users/yasminebenhamadi/YellowHammer/Training_set_YH_songs/Negative_samples_241016/"
        audio_clips(bioneg_folder, negative_folder, suffix="bioneg_", clipsize=0.5, maxclips=-1, augment=False, dir_exist_ok=True)

if __name__ == "__main__":
    positive_folder = "/Users/yasminebenhamadi/YellowHammer/Data/positive/"
    negative_folder = "/Users/yasminebenhamadi/YellowHammer/Data/negative/"

    # Clipping and augmenting the data
    prepare(positive_folder, negative_folder)

    # Preprocessing the data
    process_folder_all = "/Users/yasminebenhamadi/YellowHammer/Processed/"
    os.makedirs(process_folder_all, exist_ok = True)
    folds_file = process_folder_all+'folds.csv'
    if not os.path.isfile(folds_file):
        folds_spilt(process_folder_all)
    ids_folds = pd.read_csv(folds_file).to_numpy()

    # For NN 1
    process_folder = process_folder_all+"average/"
    if not os.path.isdir(process_folder):
        band_ranges_avg = [(i,i+1000) for i in range(3000, 10000, 1000)]
        avg_process = partial(load_process_avg, band_ranges=band_ranges_avg)
        process_folds_data(avg_process, positive_folder, negative_folder, process_folder, ids_folds)
    
    # For NN 2
    process_folder = process_folder_all+"envelope/"
    if not os.path.isdir(process_folder):
        envelope_process = partial(load_process_envelope, f_min=4000, f_max=9000, frame=1048, hop=128)
        process_folds_data(envelope_process, positive_folder, negative_folder, process_folder, ids_folds)

    # For NN bands
    process_folder = process_folder_all+"bands/"
    if not os.path.isdir(process_folder):
        band_ranges_2 = [(3500, 5500), (5000, 7000), (7000, 9000)]
        bands_process = partial(load_process_bands, band_ranges=band_ranges_2)
        process_folds_data(bands_process, positive_folder, negative_folder, process_folder, ids_folds)

    # For NN mels
    process_folder = process_folder_all+"mels/"
    if not os.path.isdir(process_folder):
        mel_process = partial(load_process_mel, nb_mels=10, f_min=3000, f_max=9000, hop=16, nfft=2048)
        process_folds_data(mel_process, positive_folder, negative_folder, process_folder, ids_folds)


    




