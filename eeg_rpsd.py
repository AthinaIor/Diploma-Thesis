import pandas as pd
import mne
import yasa
import os
from pandas import DataFrame
from scipy.signal import welch

basepath = r'C:\Users\hp\OneDrive\Υπολογιστής\ΤΗΜΜΥ\Διπλωματική\DATASETS_OFFICIAL\DATASET2_6SEC\final'
database = pd.DataFrame()

#Calculate Power Spectral Density of EEG bands (PSD)
def calculate_psd(basepath, entry):

    print("Calculating relative power spectrum of file :", entry)
    # Load data as a MNE Raw file
    raw = mne.io.read_raw_edf(basepath + "\\" + entry, preload=True, verbose=0)

    # Keep only the EEG channels
    raw.pick_types(eeg=True)

    # Extract the data and convert from V to uV
    data = raw.get_data(units="uV")
    sf = raw.info['sfreq']
    channels = raw.ch_names

    # Have a look at the data
    print('Chan =', channels)
    print('Sampling frequency =', sf, 'Hz')
    print('Data shape =', data.shape)


    win = int(4 * sf)  # Window size is set to 4 seconds
    freqs, psd = welch(data, sf, nperseg=win, average='median')

    print(freqs.shape, psd.shape)  # psd has shape (n_channels, n_frequencies)

    # Relative bandpower per channel on the whole recording (entire data)
    bp: DataFrame = yasa.bandpower(data, sf=sf, ch_names=channels, bands=[
        (0.5, 4, "Delta"),
        (4, 8, "Theta"),
        (8, 13, "Alpha"),
        (13, 35, "Beta"),
        (35, 60, "Gamma"),
    ])
    print(bp)
    return bp

#Read every file in EEG database directory for processing
for entry in os.listdir(basepath):
    if os.path.isfile(os.path.join(basepath, entry)):
        psd_data = calculate_psd(basepath, entry)
        psd_data = psd_data.unstack().to_frame().T
        psd_data.columns = psd_data.columns.map('{0[0]}_{0[1]}'.format)
        filename = entry
        psd_data['data file'] = filename
    database = database.append(psd_data)

# convert and save as csv file the power spectral density of files.
database.to_csv(r"C:\Users\hp\OneDrive\Υπολογιστής\relative_power.csv")


