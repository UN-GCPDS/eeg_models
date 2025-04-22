import numpy as np
from typing import Sequence, Tuple
from scipy.signal import resample

def load_GIGA_MI_ME(db,
              sbj: int,
              eeg_ch_names: Sequence[str],
              fs: float, 
              f_bank: np.ndarray, 
              vwt: np.ndarray, 
              new_fs: float) -> Tuple[np.ndarray, np.ndarray]:

  index_eeg_chs = db.format_channels_selectors(channels = eeg_ch_names) - 1

  tf_repr = TimeFrequencyRpr(sfreq = fs, f_bank = f_bank, vwt = vwt)

  db.load_subject(sbj)
  X, y = db.get_data(classes = ['left hand mi', 'right hand mi']) #Load MI classes, all channels {EEG}, reject bad trials, uV
  X = X[:, index_eeg_chs, :] #spatial rearrangement
  X = np.squeeze(tf_repr.transform(X))
  #Resampling
  if new_fs == fs:
    print('No resampling, since new sampling rate same.')
  else:
    print("Resampling from {:f} to {:f} Hz.".format(fs, new_fs))
    X = resample(X, int((X.shape[-1]/fs)*new_fs), axis = -1)
    
  return X, y


  #######################

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import clear_session
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    roc_auc_score,
    f1_score,
    recall_score,
    precision_score
)


def train(db_name, load_args, cv_args, model_args, compile_args, fit_args, seed):
    X_train, y_train = load_DB(db_name, **load_args)
    X_train = X_train[..., np.newaxis]
    
    cv_results = {'params': [],
                  'mean_acc': np.zeros(cv_args['cv'].get_n_splits()),
                  'mean_kappa': np.zeros(cv_args['cv'].get_n_splits()),
                  'mean_auc': np.zeros(cv_args['cv'].get_n_splits()),
                  'mean_f1_left': np.zeros(cv_args['cv'].get_n_splits()),
                  'mean_f1_right': np.zeros(cv_args['cv'].get_n_splits()),
                  'mean_recall_left': np.zeros(cv_args['cv'].get_n_splits()),
                  'mean_recall_right': np.zeros(cv_args['cv'].get_n_splits()),
                  'mean_precision_left': np.zeros(cv_args['cv'].get_n_splits()),
                  'mean_precision_right': np.zeros(cv_args['cv'].get_n_splits()),}
    
    if model_args['nb_classes'] == 4:
        cv_results['mean_f1_legs'] = np.zeros(cv_args['cv'].get_n_splits())
        cv_results['mean_f1_tongue'] = np.zeros(cv_args['cv'].get_n_splits())
        cv_results['mean_recall_legs'] = np.zeros(cv_args['cv'].get_n_splits())
        cv_results['mean_recall_tongue'] = np.zeros(cv_args['cv'].get_n_splits())
        cv_results['mean_precision_legs'] = np.zeros(cv_args['cv'].get_n_splits())
        cv_results['mean_precision_tongue'] = np.zeros(cv_args['cv'].get_n_splits())

    k = 0
    max_acc = -np.inf
    for train_index, val_index in cv_args['cv'].split(X_train, y_train):
      X, X_val = X_train[train_index], X_train[val_index]
      y, y_val = y_train[train_index], y_train[val_index]
      print(val_index)

      if model_args['autoencoder']:
        y = [X, y]

      batch_size, C, T = X.shape[:-1]

      clear_session()
      set_seed(seed)

      model_cll, model_params = get_model(model_args['model_name'], model_args['nb_classes'])
      model = model_cll(**model_params, Chans = C, Samples = T)
      model.compile(loss = compile_args['loss'], 
                    optimizer = Adam(compile_args['init_lr']))
      
      history = model.fit(X, y,
                batch_size = batch_size,
                **fit_args)

      if model_args['autoencoder']:
        y_prob = model.predict(X_val)[-1]
        y_pred = np.argmax(y_prob, axis = 1)
      else:
        y_prob = model.predict(X_val)
        y_pred = np.argmax(y_prob, axis = 1)

      cv_results['mean_acc'][k] = accuracy_score(y_val, y_pred)
      cv_results['mean_kappa'][k] = cohen_kappa_score(y_val, y_pred)
      if model_args['nb_classes'] == 2:
        cv_results['mean_auc'][k] = roc_auc_score(y_val, y_prob[:, 1], average = 'macro')
        cv_results['mean_f1_left'][k] = f1_score(y_val, y_pred, pos_label = 0, average = 'binary')
        cv_results['mean_f1_right'][k] = f1_score(y_val, y_pred, pos_label = 1, average = 'binary')
        cv_results['mean_recall_left'][k] = recall_score(y_val, y_pred, pos_label = 0, average = 'binary')
        cv_results['mean_recall_right'][k] = recall_score(y_val, y_pred, pos_label = 1, average = 'binary')
        cv_results['mean_precision_left'][k] = precision_score(y_val, y_pred, pos_label = 0, average = 'binary')
        cv_results['mean_precision_right'][k] = precision_score(y_val, y_pred, pos_label = 1, average = 'binary')
      else:                                                                                  
        cv_results['mean_auc'][k] = roc_auc_score(y_val, y_prob, average = 'macro', multi_class = 'ovo')
        
        cv_results['mean_f1_left'][k] = f1_score(y_val, y_pred, pos_label = 0, average = 'micro')
        cv_results['mean_f1_right'][k] = f1_score(y_val, y_pred, pos_label = 1, average = 'micro')
        cv_results['mean_f1_legs'][k] = f1_score(y_val, y_pred, pos_label = 2, average = 'micro')
        cv_results['mean_f1_tongue'][k] = f1_score(y_val, y_pred, pos_label = 3, average = 'micro')
        cv_results['mean_recall_left'][k] = recall_score(y_val, y_pred, pos_label = 0, average = 'micro')
        cv_results['mean_recall_right'][k] = recall_score(y_val, y_pred, pos_label = 1, average = 'micro')
        cv_results['mean_recall_legs'][k] = recall_score(y_val, y_pred, pos_label = 2, average = 'micro')
        cv_results['mean_recall_tongue'][k] = recall_score(y_val, y_pred, pos_label = 3, average = 'micro')
        cv_results['mean_precision_left'][k] = precision_score(y_val, y_pred, pos_label = 0, average = 'micro')
        cv_results['mean_precision_right'][k] = precision_score(y_val, y_pred, pos_label = 1, average = 'micro')
        cv_results['mean_precision_legs'][k] = precision_score(y_val, y_pred, pos_label = 2, average = 'micro')
        cv_results['mean_precision_tongue'][k] = precision_score(y_val, y_pred, pos_label = 3, average = 'micro')
                                                       
                                                       
      if cv_results['mean_acc'][k]  > max_acc:
        max_acc = cv_results['mean_acc'][k]
        model.save_weights('sbj' + str(load_args['sbj']) +'.h5')

      k += 1
                                                
    cv_results['std_acc'] = round(cv_results['mean_acc'].std(), 3)
    cv_results['mean_acc'] = round(cv_results['mean_acc'].mean(), 3)
    cv_results['std_kappa'] = round(cv_results['mean_kappa'].std(), 3)
    cv_results['mean_kappa'] = round(cv_results['mean_kappa'].mean(), 3)
    cv_results['std_auc'] = round(cv_results['mean_auc'].std(), 3)
    cv_results['mean_auc'] = round(cv_results['mean_auc'].mean(), 3)
      
    cv_results['mean_f1_left'] = round(cv_results['mean_f1_left'].mean(), 3)
    cv_results['std_f1_left'] = round(cv_results['mean_f1_left'].std(), 3)
    cv_results['mean_f1_right'] = round(cv_results['mean_f1_right'].mean(), 3)
    cv_results['std_f1_right'] = round(cv_results['mean_f1_right'].std(), 3)
    cv_results['mean_recall_left'] = round(cv_results['mean_recall_left'].mean(), 3)
    cv_results['std_recall_left'] = round(cv_results['mean_recall_left'].std(), 3)
    cv_results['mean_recall_right'] = round(cv_results['mean_recall_right'].mean(), 3)
    cv_results['std_recall_right'] = round(cv_results['mean_recall_right'].std(), 3)
    cv_results['mean_precision_left'] = round(cv_results['mean_precision_left'].mean(), 3)
    cv_results['std_precision_left'] = round(cv_results['mean_precision_left'].std(), 3)
    cv_results['mean_precision_right'] = round(cv_results['mean_precision_right'].mean(), 3)
    cv_results['std_precision_right'] = round(cv_results['mean_precision_right'].std(), 3)

    if model_args['nb_classes'] == 4:
        cv_results['mean_f1_legs'] = round(cv_results['mean_f1_legs'].mean(), 3)
        cv_results['std_f1_legs'] = round(cv_results['mean_f1_legs'].std(), 3)
        cv_results['mean_f1_tongue'] = round(cv_results['mean_f1_tongue'].mean(), 3)
        cv_results['std_f1_tongue'] = round(cv_results['mean_f1_tongue'].std(), 3)
        cv_results['mean_recall_legs'] = round(cv_results['mean_recall_legs'].mean(), 3)
        cv_results['std_recall_legs'] = round(cv_results['mean_recall_legs'].std(), 3)
        cv_results['mean_recall_tongue'] = round(cv_results['mean_recall_tongue'].mean(), 3)
        cv_results['std_recall_tongue'] = round(cv_results['mean_recall_tongue'].std(), 3)
        cv_results['mean_precision_legs'] = round(cv_results['mean_precision_legs'].mean(), 3)
        cv_results['std_precision_legs'] = round(cv_results['mean_precision_legs'].std(), 3)
        cv_results['mean_precision_tongue'] = round(cv_results['mean_precision_tongue'].mean(), 3)
        cv_results['std_precision_tongue'] = round(cv_results['mean_precision_tongue'].std(), 3)
    
    return cv_results