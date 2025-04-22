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