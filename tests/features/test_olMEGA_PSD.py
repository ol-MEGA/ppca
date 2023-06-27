import torch
import math
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing import features
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#signal = read_audio('tests/samples/single-mic/example1.wav')
#signal = signal.unsqueeze(0)
sample_rate = 16000
signal = torch.zeros([1, sample_rate])
signal[0, 0] = 1

compute_STFT = features.STFT(
         sample_rate=sample_rate, win_length=25, hop_length=12.5, n_fft=512
    )
STFT = compute_STFT(signal)
print(STFT.shape)

psd = features.spectral_magnitude(STFT)
print(psd.shape)

psd_smoothed = features.smooth_power(psd)
print(psd_smoothed.shape)


# plot
fig, ax = plt.subplots(3, 1, figsize=(5,6))

ax[0].plot(signal.squeeze())
ax[0].set_xlabel("time in samples")
ax[0].set_ylabel("amplitude")

im = ax[1].imshow(10*torch.log10(psd.squeeze().t()), extent=[0, psd.shape[1], 0, sample_rate/2], origin='lower', interpolation='nearest')
ax[1].set_xlabel("time frame")
ax[1].set_ylabel("freq in Hz")
ax[1].set_aspect('auto')
ax[1].set_aspect('auto')
divider = make_axes_locatable(ax[1])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('PSD in dB')
im.set_clim(-50, 0)

im = ax[2].imshow(10*torch.log10(psd_smoothed.squeeze().t()), extent=[0, psd_smoothed.shape[1], 0, sample_rate/2], origin='lower', interpolation='nearest')
ax[2].set_xlabel("time frame")
ax[2].set_ylabel("freq in Hz")
ax[2].set_aspect('auto')
divider = make_axes_locatable(ax[2])
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('PSD smoothed in dB')
im.set_clim(-50, 0)
ax2 = ax[2].twinx()
ax2.plot(psd_smoothed.squeeze()[:, 0].t(), 'k')

plt.tight_layout()
plt.savefig('test_impulse_olMEGA.png')
plt.show()
