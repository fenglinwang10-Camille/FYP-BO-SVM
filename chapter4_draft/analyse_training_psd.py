"""Descriptive PSD of the saved main-experiment training partition only."""
import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_loader import _infer_class_from_path, _load_signal_file

OUT = ROOT / 'chapter4_draft' / 'training_psd'
OUT.mkdir(parents=True, exist_ok=True)
manifest_path = ROOT / 'results/09_fixed_fold_segment_accuracy/seed_42/uniform/grouped_split_manifest.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
train = manifest['train_segment_ids']
test = manifest['test_segment_ids']
assert len(train) == 400 and len(test) == 100
assert not set(train).intersection(test)
dataset = ROOT / 'Dataset'
files = sorted(p for p in dataset.rglob('*') if p.is_file() and p.suffix.lower() in {'.txt', '.asc', '.csv'})
files = [p for p in files if _infer_class_from_path(p, dataset) is not None]
assert len(files) == 500
names = ['Z', 'O', 'N', 'F', 'S']
settings = dict(fs=173.61, window='hann', nperseg=1024, noverlap=512,
                nfft=1024, detrend='constant', return_onesided=True, scaling='density', average='mean')
rows, spectra, labels = [], [], []
for idx in train:
    path = files[idx]
    signal = _load_signal_file(path)
    assert len(signal) == 4097 and np.isfinite(signal).all()
    label = _infer_class_from_path(path, dataset)
    freq, psd = welch(signal, **settings)
    spectra.append(psd)
    labels.append(label)
    rows.append(dict(recording_id=idx, label=names[label], file=str(path.relative_to(ROOT)),
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
spectra, labels = np.array(spectra), np.array(labels)
assert all(np.sum(labels == i) == 80 for i in range(5))
pd.DataFrame(rows).to_csv(OUT / 'training_recordings.csv', index=False)
pd.DataFrame(spectra.T, index=freq, columns=train).rename_axis('frequency_hz').to_csv(OUT / 'per_recording_psd.csv')
averages = {name: spectra[labels == i].mean(axis=0) for i, name in enumerate(names)}
pd.DataFrame({'frequency_hz': freq, **averages}).to_csv(OUT / 'class_mean_psd.csv', index=False)

def area(y, low, high):
    inside = (freq > low) & (freq < high)
    x = np.r_[low, freq[inside], high]
    values = np.interp(x, freq, y)
    return float(np.trapz(values, x))

summary = []
for name, psd in averages.items():
    total = area(psd, 0, settings['fs']/2)
    peak_mask = (freq >= 0.5) & (freq <= 40)
    summary.append(dict(class_label=name, recordings=80,
        peak_frequency_0_5_to_40_hz=float(freq[peak_mask][np.argmax(psd[peak_mask])]),
        fraction_0_5_to_30=area(psd, 0.5, 30)/total,
        fraction_30_to_60=area(psd, 30, 60)/total,
        fraction_above_60=area(psd, 60, settings['fs']/2)/total))
pd.DataFrame(summary).to_csv(OUT / 'spectral_summary.csv', index=False)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
fig, ax = plt.subplots(figsize=(8, 3.8), layout='constrained')
colors = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#D55E00']
for (name, psd), color in zip(averages.items(), colors):
    ax.semilogy(freq, psd, color=color, linewidth=1.6, linestyle='-', label=name)
ax.set(xlim=(0, settings['fs']/2), xlabel='Frequency (Hz)',
       ylabel='Mean PSD (signal units²/Hz)')
ax.legend(title='Class (80 training recordings each)', ncol=5, loc='upper right', frameon=False)
ax.grid(axis='y', which='major', alpha=.2)
ax.spines[['top', 'right']].set_visible(False)
for suffix in ['png', 'pdf']:
    fig.savefig(OUT / f'training_class_mean_psd.{suffix}', dpi=300)
plt.close(fig)
protocol = dict(split_source=str(manifest_path.relative_to(ROOT)),
    split_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    training_recordings=400, test_signals_read=False, counts={n:80 for n in names},
    preprocessing='None; Welch subtracts each spectral segment mean (constant detrending).',
    welch=settings, averaging='Arithmetic mean across recording PSDs before logarithmic plotting.',
    frequency_resolution_hz=settings['fs']/1024,
    summary='Integrals of class-mean PSD, with linear interpolation at interval endpoints; fractions of 0-Nyquist power.',
    purpose='Post-search descriptive analysis; does not establish optimal cutoff values.',
    difference_from_features='1024 samples for full-recording illustration; classifier feature Welch uses min(256, window length).')
(OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
(OUT / 'figure_insert.tex').write_text(r'''\begin{figure}[htbp]
    \centering
    \includegraphics[width=\textwidth]{figures/training_class_mean_psd.pdf}
    \caption{Class-wise mean Welch PSD of the unfiltered training recordings
    from the main experiment (seed 42; 80 recordings per class).
    Each recording is analysed using a 1024-sample Hann window with
    50\% overlap and constant detrending. Recording PSDs are averaged
    before plotting on a logarithmic vertical axis.}
    \label{fig:training-psd}
\end{figure}
''', encoding='utf-8')
print(pd.DataFrame(summary).to_string(index=False))
print('Outputs:', OUT)
