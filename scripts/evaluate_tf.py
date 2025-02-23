

from evaluate import *
from read_data import *


def get_tf_results (data_folder, fold, model_path, transpose):
    # TODO
    X_test, y_test, names = get_folds_data(data_folder, nb_fold=fold, data_name='test', transpose=transpose, augment=False)
    model = tf.keras.models.load_model(model_path)
    y_pred_scores = model.predict(X_test).flatten()
    y_pred = (y_pred_scores>0.5).astype(int).flatten()
    return y_test, y_pred, y_pred_scores, names