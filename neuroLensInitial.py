import os
import argparse
import sys
import numpy as np
import scipy.signal as signal
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

try:
    import mne
    HAVE_MNE = True
except Exception:
    HAVE_MNE = False

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    HAVE_TF = True
except Exception:
    HAVE_TF = False


def loadEeg(path, channels=None, sfreq=None):
    """
    Load EEG data. Supports EDF (if mne available), CSV, or NumPy.

    Returns:
        data: ndarray (nChannels, nSamples)
        sfreq: sampling frequency (Hz)
        chNames: list of channel names
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.edf', '.bdf') and HAVE_MNE:
        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
        if channels is not None:
            raw.pick_channels(channels)
        data = raw.get_data()
        sfreq = int(raw.info['sfreq'])
        chNames = raw.ch_names
        return data, sfreq, chNames

    try:
        arr = np.load(path)
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr, sfreq or 256.0, [f'ch{i}' for i in range(arr.shape[0])]
    except Exception:
        pass

    try:
        import pandas as pd
        df = pd.read_csv(path, header=0)
        arr = df.values
        if arr.ndim != 2:
            arr = np.atleast_2d(arr)

        if arr.shape[0] > arr.shape[1]:
            arr = arr.T
            chNames = list(df.columns) if df.columns is not None else [f'ch{i}' for i in range(arr.shape[0])]
        else:
            if hasattr(df, 'index') and not isinstance(df.index, pd.RangeIndex):
                chNames = [str(i) for i in df.index]
            else:
                chNames = [f'ch{i}' for i in range(arr.shape[0])]
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr, sfreq or 256.0, chNames
    except Exception:
        pass

    try:
        arr = np.loadtxt(path, delimiter=',')
        if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
            arr = arr.T
        if arr.ndim == 1:
            arr = arr[np.newaxis, :]
        return arr, sfreq or 256.0, [f'ch{i}' for i in range(arr.shape[0])]
    except Exception:
        raise ValueError('Could not load EEG file. Install mne for EDF support or provide a valid CSV/NPY file.')


def bandpassFilter(data, sfreq, lFreq=1.0, hFreq=50.0, order=4):
    nyq = 0.5 * sfreq
    low = lFreq / nyq
    high = hFreq / nyq
    low = max(1e-6, min(low, 0.999))
    high = max(1e-6, min(high, 0.999))
    if low >= high:
        raise ValueError('Invalid bandpass frequencies: low must be < high after normalization')
    b, a = signal.butter(order, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, data, axis=-1)
    return filtered


def notchFilter(data, sfreq, notchFreq=50.0, quality=30.0):
    if notchFreq <= 0 or sfreq <= 0:
        return data
    w0 = notchFreq / (sfreq / 2)
    if not (0 < w0 < 1):
        return data
    b, a = signal.iirnotch(w0, quality)
    return signal.filtfilt(b, a, data, axis=-1)


def epochData(data, sfreq, epochSec=2.0, overlap=0.5):
    nSamples = int(data.shape[1])
    step = int(max(1, round(epochSec * sfreq * (1 - overlap))))
    win = int(max(1, round(epochSec * sfreq)))
    epochs = []
    indices = []
    for start in range(0, nSamples - win + 1, step):
        epochs.append(data[:, start:start+win])
        indices.append((start, start+win))
    if len(epochs) == 0:
        pad = np.zeros((data.shape[0], win))
        pad[:, :data.shape[1]] = data
        epochs = [pad]
        indices = [(0, data.shape[1])]
    return np.stack(epochs, axis=0), indices


def computeBandPowers(epoch, sfreq, bands=None):
    """
    Compute relative band powers for an epoch per channel.

    epoch: (nChan, nSamples)
    returns: (nChan, nBands)
    """
    if bands is None:
        bands = {'delta': (1, 4), 'theta': (4, 8), 'alpha': (8, 13), 'beta': (13, 30), 'gamma': (30, 45)}
    nChan = epoch.shape[0]
    nperseg = min(256, epoch.shape[1])
    psdFreqs, psd = signal.welch(epoch, fs=sfreq, nperseg=nperseg, axis=-1)
    bandPowers = np.zeros((nChan, len(bands)))
    totalPower = np.trapz(psd, psdFreqs, axis=-1)
    for i, (name, (fmin, fmax)) in enumerate(bands.items()):
        idx = np.logical_and(psdFreqs >= fmin, psdFreqs <= fmax)
        if np.sum(idx) == 0:
            p = np.zeros(psd.shape[0])
        else:
            p = np.trapz(psd[..., idx], psdFreqs[idx], axis=-1)
        bandPowers[:, i] = p / (totalPower + 1e-12)
    return bandPowers, list(bands.keys())


def hjorthParameters(x):
    """x: 1D signal. Returns Activity, Mobility, and Complexity."""
    if len(x) < 3:
        return 0.0, 0.0, 0.0
    firstDeriv = np.diff(x)
    secondDeriv = np.diff(firstDeriv)
    var0 = np.var(x)
    var1 = np.var(firstDeriv)
    var2 = np.var(secondDeriv)
    activity = var0
    mobility = np.sqrt(var1 / (var0 + 1e-12))
    complexity = np.sqrt(var2 / (var1 + 1e-12)) / (mobility + 1e-12)
    return activity, mobility, complexity


def spectralEntropy(x, sfreq, nperseg=256):
    f, Pxx = signal.welch(x, fs=sfreq, nperseg=min(nperseg, len(x)))
    Pxx = Pxx + 1e-12
    Pnorm = Pxx / np.sum(Pxx)
    se = -np.sum(Pnorm * np.log2(Pnorm))
    return se


def extractFeatures(epoch, sfreq):
    """
    Extract features for a single epoch (nChan, nSamples).
    Returns a 1D vector of concatenated channel-wise features.
    Features per channel: band powers (5), hjorth (3), spectral entropy (1), mean, std (2) -> 11 total
    """
    bandPowers, bandNames = computeBandPowers(epoch, sfreq)
    nChan = epoch.shape[0]
    feats = []
    for ch in range(nChan):
        chEpoch = epoch[ch]
        bp = bandPowers[ch]
        hj = hjorthParameters(chEpoch)
        se = spectralEntropy(chEpoch, sfreq)
        mean = np.mean(chEpoch)
        sd = np.std(chEpoch)
        feats.extend(bp.tolist())
        feats.extend(hj)
        feats.append(se)
        feats.append(mean)
        feats.append(sd)
    return np.array(feats)


def detectAnomaliesIsolationForest(allFeatureVectors, contamination=0.01, randomState=42):
    """Robust IsolationForest wrapper that handles NaN/inf and constant columns."""
    X = np.array(allFeatureVectors, dtype=float)
    X[np.isposinf(X)] = np.nan
    X[np.isneginf(X)] = np.nan
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    stds = X.std(axis=0)
    nonconst_idx = np.where(stds > 1e-12)[0]
    if len(nonconst_idx) < X.shape[1]:
        X = X[:, nonconst_idx]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    iso = IsolationForest(contamination=contamination, random_state=randomState)
    iso.fit(Xs)
    scores = iso.decision_function(Xs)
    preds = iso.predict(Xs)
    return preds, scores, scaler, iso


def buildAutoencoder(inputDim, latentDim=32):
    if not HAVE_TF:
        raise RuntimeError('TensorFlow/Keras not available for autoencoder.')
    inputs = layers.Input(shape=(inputDim,))
    x = layers.Dense(max(4, int(inputDim/2)), activation='relu')(inputs)
    x = layers.Dense(max(4, int(inputDim/4)), activation='relu')(x)
    latent = layers.Dense(latentDim, activation='relu')(x)
    x = layers.Dense(max(4, int(inputDim/4)), activation='relu')(latent)
    x = layers.Dense(max(4, int(inputDim/2)), activation='relu')(x)
    outputs = layers.Dense(inputDim, activation='linear')(x)
    model = keras.Model(inputs, outputs, name='autoencoder')
    model.compile(optimizer='adam', loss='mse')
    return model


def detectAnomaliesAutoencoder(allFeatureVectors, contamination=0.01, epochs=50, batchSize=32):
    if not HAVE_TF:
        raise RuntimeError('TensorFlow/Keras not available for autoencoder.')
    X = np.array(allFeatureVectors, dtype=float)
    X[np.isposinf(X)] = np.nan
    X[np.isneginf(X)] = np.nan
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    stds = X.std(axis=0)
    nonconst_idx = np.where(stds > 1e-12)[0]
    if len(nonconst_idx) < X.shape[1]:
        X = X[:, nonconst_idx]

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    inputDim = Xs.shape[1]
    model = buildAutoencoder(inputDim)
    model.fit(Xs, Xs, epochs=epochs, batch_size=batchSize, verbose=0)
    recon = model.predict(Xs, verbose=0)
    mse = np.mean((Xs - recon)**2, axis=1)
    thresh = np.percentile(mse, 100 * (1 - contamination))
    preds = np.where(mse > thresh, -1, 1)
    return preds, mse, scaler, model, thresh


def plotEpochWithAnomaly(epoch, sfreq, chNames=None, title='Epoch'):
    nChan, nSamples = epoch.shape
    t = np.arange(nSamples) / sfreq
    maxRows = 12
    rows = min(nChan, maxRows)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 2*rows), sharex=True)
    if rows == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, epoch[i])
        name = chNames[i] if chNames is not None and i < len(chNames) else f'Ch{i}'
        ax.set_ylabel(name)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def runPipeline(path, method='isolation_forest', sfreq=None, channels=None, epochSec=2.0, overlap=0.5, contamination=0.01, useNotch=True, notchFreq=50.0, ampThresh=200e-6):
    data, sfreqLoaded, chNames = loadEeg(path, channels=channels, sfreq=sfreq)
    sfreq = float(sfreq or sfreqLoaded)
    print(f'Loaded data: channels={data.shape[0]}, samples={data.shape[1]}, sfreq={sfreq}')

    # Filtering
    print('Applying bandpass filter (1-50 Hz)')
    filtered = bandpassFilter(data, sfreq, 1.0, 50.0)
    if useNotch:
        print(f'Applying notch filter at {notchFreq} Hz')
        filtered = notchFilter(filtered, sfreq, notchFreq=notchFreq)

    epochs, indices = epochData(filtered, sfreq, epochSec=epochSec, overlap=overlap)
    nEpochs = int(epochs.shape[0])
    print(f'Created {nEpochs} epochs of {epochSec}s with {overlap*100:.0f}% overlap')

    cleanEpochs_list = []
    cleanIndices = []
    for i in range(nEpochs):
        ep = epochs[i]
        if np.isnan(ep).any():
            continue
        if np.nanmax(np.abs(ep)) > ampThresh:
            continue
        cleanEpochs_list.append(ep)
        cleanIndices.append(indices[i])
    if len(cleanEpochs_list) == 0:
        print('Warning: all epochs rejected by amplitude threshold. Using all epochs (lower threshold?)')
        cleanEpochs = epochs
        cleanIndices = indices
    else:
        cleanEpochs = np.stack(cleanEpochs_list, axis=0)
    print(f'{cleanEpochs.shape[0]} epochs remain after amplitude-based rejection')

    featList = []
    for i in range(cleanEpochs.shape[0]):
        feat = extractFeatures(cleanEpochs[i], sfreq)
        featList.append(feat)
    X = np.stack(featList, axis=0)
    print('Extracted feature matrix shape:', X.shape)

    if method == 'isolation_forest':
        preds, scores, scaler, model = detectAnomaliesIsolationForest(X, contamination=contamination)
        anomalyIdx = np.where(preds == -1)[0]
        print(f'Found {len(anomalyIdx)} anomalous epochs (IsolationForest)')
        for idx in anomalyIdx[:5]:
            start, end = cleanIndices[idx]
            title = f'Anomalous epoch #{idx} (samples {start}:{end})'
            plotEpochWithAnomaly(cleanEpochs[idx], sfreq, chNames=chNames, title=title)
        return {'method': 'isolation_forest', 'preds': preds, 'scores': scores, 'indices': cleanIndices}

    elif method == 'autoencoder':
        preds, mse, scaler, model, thresh = detectAnomaliesAutoencoder(X, contamination=contamination)
        anomalyIdx = np.where(preds == -1)[0]
        print(f'Found {len(anomalyIdx)} anomalous epochs (Autoencoder)')
        for idx in anomalyIdx[:5]:
            start, end = cleanIndices[idx]
            title = f'Anomalous epoch #{idx} (samples {start}:{end}), reconMSE={mse[idx]:.4e}'
            plotEpochWithAnomaly(cleanEpochs[idx], sfreq, chNames=chNames, title=title)
        return {'method': 'autoencoder', 'preds': preds, 'mse': mse, 'indices': cleanIndices, 'threshold': thresh}

    else:
        raise ValueError('Unknown method: choose isolation_forest or autoencoder')

if __name__ == '__main__':
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='EEG anomaly detection')
        parser.add_argument('--input', required=True, help='Path to EEG file (.edf, .npy, .csv)')
        parser.add_argument('--method', default='isolation_forest', choices=['isolation_forest', 'autoencoder'])
        parser.add_argument('--sfreq', type=float, default=None)
        parser.add_argument('--epochSec', type=float, default=2.0)
        parser.add_argument('--overlap', type=float, default=0.5)
        parser.add_argument('--contamination', type=float, default=0.01)
        parser.add_argument('--notchFreq', type=float, default=50.0)
        parser.add_argument('--ampThresh', type=float, default=200e-6)
        args = parser.parse_args()
        res = runPipeline(args.input, method=args.method, sfreq=args.sfreq, epochSec=args.epochSec, overlap=args.overlap, contamination=args.contamination, notchFreq=args.notchFreq, ampThresh=args.ampThresh)
        print('Done')
    else:
        print('No CLI arguments detected — running example pipeline (Spyder/interactive mode).')
        example_path = 'features_raw.csv'
        if not os.path.exists(example_path):
            print(f"Example file '{example_path}' not found. Import this module and call runPipeline(path, ...) manually.")
        else:
            res = runPipeline(example_path, method='isolation_forest')
            print('Done')
