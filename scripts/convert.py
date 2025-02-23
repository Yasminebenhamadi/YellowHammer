
import os
import numpy as np
import tensorflow as tf

from read_data import *

def convert_model(outfolder, model_path, name, data_folder, fold, transpose, keep_data):
    # Code taken from students
    model = tf.keras.models.load_model(model_path)

    X_train, y_train, names = get_folds_data(data_folder, nb_fold=fold, data_name='train', transpose=transpose, augment=False)

    # Load sample training data (adjust shape to match model input)
    def representative_dataset(size=100, keep=keep_data):
        if not keep:
            X_train_reshaped = X_train[..., np.newaxis]
        else:
            X_train_reshaped = X_train
        for i in range(size):
            sample = X_train_reshaped[i]  # Take one sample
            sample = np.expand_dims(sample, axis=0)  # Add batch dimension
            yield [sample.astype(np.float32)] # Convert to float32

    # Convert to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model_1 = converter.convert()

    # Save the converted model to a .tflite file
    with open(outfolder+name+'_trained_model.tflite', 'wb') as f:
        f.write(tflite_model_1)

    # Enable full integer quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset

    # Specify full int8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # Convert and save
    tflite_model = converter.convert()

    with open(outfolder+name+"_quantized_model.tflite", "wb") as f:
        f.write(tflite_model)

def convert_models():
    data_folder_all = "/Users/yasminebenhamadi/YellowHammer/Processed/"
    models_folder_all = "/Users/yasminebenhamadi/YellowHammer/models/"
    tf_folder = "/Users/yasminebenhamadi/YellowHammer/TFLite/"

    process_names = ["average", "envelope", "bands", "mels"]

    data_folders = [data_folder_all+process_name+"/" for process_name in process_names]
    models_folders = [models_folder_all+process_name+"_models/" for process_name in process_names]

    for fld_nb in range(1):
        tflite_folder_fld=tf_folder+"fold_"+str(fld_nb)+"/"
        os.makedirs(tflite_folder_fld, exist_ok=True)

        fld_models_folders = [models_folders[i]+process_names[i]+"_model_"+str(fld_nb)+"/model.h5" for i in range(len(process_names))]
        

        for i in range(len(process_names)):
            convert_model(tflite_folder_fld, model_path=fld_models_folders[i], name=process_names[i], data_folder=data_folders[i], fold=fld_nb, transpose=(i==3), keep_data=(i==3))

if __name__ == "__main__":
    convert_models()





