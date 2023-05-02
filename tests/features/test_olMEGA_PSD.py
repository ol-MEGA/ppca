import torch
import math
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing import features
import matplotlib.pyplot as plt

signal = read_audio('tests/samples/single-mic/example1.wav')
signal = signal.unsqueeze(0)
print(signal.shape)
sample_rate = 16000

compute_STFT = features.STFT(
         sample_rate=sample_rate, win_length=25, hop_length=12.5, n_fft=512
    )
STFT = compute_STFT(signal)
print(STFT.shape)

mag = features.spectral_magnitude(STFT)
print(mag.shape)

mag_smoothed = features.smooth_magnitude(mag)
print(mag_smoothed.shape)


# plot
fig, ax = plt.subplots(3, 1, figsize=(5,6))

ax[0].plot(signal.squeeze())
ax[0].set_xlabel("time in samples")
ax[0].set_ylabel("amplitude")

im = ax[1].imshow(10*torch.log10(mag.squeeze().t()), extent=[0, mag.shape[1], 0, sample_rate/2], origin='lower', interpolation='nearest')
ax[1].set_xlabel("time frame")
ax[1].set_ylabel("freq in Hz")
ax[1].set_aspect('auto')
im.set_clim(-80, 0)

im = ax[2].imshow(10*torch.log10(mag_smoothed.squeeze().t()), extent=[0, mag_smoothed.shape[1], 0, sample_rate/2], origin='lower', interpolation='nearest')
ax[2].set_xlabel("time frame")
ax[2].set_ylabel("freq in Hz")
ax[2].set_aspect('auto')
im.set_clim(-80, 0)

plt.tight_layout()
plt.show()
