
import tensorflow as tf

from evaluate import *
from read_data import *

def get_pred_details(interpreter, output_details, quantized=False):
    output_data = interpreter.get_tensor(output_details[0]['index'])

    output_scale, output_zero_point = output_details[0]['quantization']

    if quantized:
        output_scale, output_zero_point = output_details[0]['quantization']
        output_data = (output_data.astype(np.float32)-output_zero_point)*output_scale
    
    if len(output_details)==1:
        label = 1 if output_data>0.5 else 0
        return label, output_data
    elif output_data>0.9:
        return 1, output_data
    elif output_data<0.1:
        return 0, output_data
    else:
        output_data_2 = interpreter.get_tensor(output_details[1]['index'])
        if quantized:
            output_scale_2, output_zero_point_2 = output_details[1]['quantization']
            output_data_2 = (output_data_2.astype(np.float32)-output_zero_point_2)*output_scale_2

        label = 1 if output_data_2>0.5 else 0
        return label, output_data_2

def get_tf_results (data_folder, fold, model_path, transpose, expand):

    # TFLite without optimization
    X_test, y_test, names = get_folds_data(data_folder, nb_fold=fold, data_name='test', transpose=transpose, augment=False)

    if expand:
        X_test = X_test[..., np.newaxis]

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get the input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Get quantization parameters
    input_scale, input_zero_point = input_details[0]['quantization']

    y_pred=[]
    y_pred_scores=[]
    for x in X_test:
        # Normalize and quantize input to INT8
        test_input = x.astype(np.float32)  # Ensure float32 before scaling

        if "int8" in model_path:
            test_input = (test_input / input_scale + input_zero_point).astype(np.int8)

        test_input = np.expand_dims(test_input, axis=0)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()

        label, score = get_pred_details(interpreter, output_details, quantized="int8" in model_path)
        y_pred.append(label)
        y_pred_scores.append(score)

    return y_test, np.array(y_pred), np.array(y_pred_scores).flatten(), names

def eval_tf_results(data_folder, fold, model_path, transpose, expand, model_name, eval_plot_folder_fld, plot_hist=True, plot_conf=False):
    y_test, y_pred, y_pred_scores, names = get_tf_results (data_folder, fold, model_path, transpose,expand)
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

def compare_TFLite_models():
    data_folder_all = "/Users/yasminebenhamadi/YellowHammer/Processed/"
    models_folder_all = "/Users/yasminebenhamadi/YellowHammer/TFLite/"
    type_model="int8"
    eval_plot_folder = "/Users/yasminebenhamadi/YellowHammer/plots_TFLite_"+type_model+"/"

    process_names = ["average", "envelope", "bands", "mels", "mels"]
    model_names = ["average", "envelope", "bands", "mels", "exit"]

    data_folders = [data_folder_all+process_name+"/" for process_name in process_names]
    models_folders = [models_folder_all+process_name+"_models/" for process_name in process_names]

    for fld_nb in range(1):
        eval_plot_folder_fld=eval_plot_folder+"fold_"+str(fld_nb)+"/"
        os.makedirs(eval_plot_folder_fld, exist_ok=True)

        models_folder_fld = models_folder_all+"fold_"+str(fld_nb)+"/"

        model_paths = [models_folder_fld+model_names[i]+"_"+type_model+"_model.tflite" for i in range(len(model_names))]

        # Read data
        precision_list = []
        f1_list = []
        accuracy_list = []

        all_recalls = []
        all_precisions = []

        hist = True
        conf = True

        for i in range(len(model_names)):
            expand = (i==1 or i==2)
            recalls_at, precisions_at, precision, f1, accuracy = eval_tf_results(data_folders[i], fld_nb, model_paths[i], transpose=(i>=3), expand=expand, model_name=model_names[i], eval_plot_folder_fld=eval_plot_folder_fld, plot_hist=hist, plot_conf=conf)
            all_recalls.append(recalls_at)
            all_precisions.append(precisions_at)
            precision_list.append(precision)
            f1_list.append(f1)
            accuracy_list.append(accuracy)

        plot_curve_at_distance(eval_plot_folder_fld, all_recalls, model_names, "Recall at distance")
        plot_curve_at_distance(eval_plot_folder_fld, all_precisions, model_names, "Precision at distance")

        plot_ordered(eval_plot_folder_fld, precision_list, model_names, "Precision", color="green")
        plot_ordered(eval_plot_folder_fld, f1_list, model_names, "F1 score", color="red")
        plot_ordered(eval_plot_folder_fld, accuracy_list, model_names, "Accuracy", color="blue")


if __name__ == "__main__":
    compare_TFLite_models()