import torch
import math
from speechbrain.dataio.dataio import read_audio
from speechbrain.processing import features
import matplotlib.pyplot as plt

def smooth_magnitude(mag, hop_len=12.5, tau=125):
    """Returns the smoothed magnitude with smoothing time tau
    by using a first order low-pass filter.

    Arguments
    ---------
    mag : torch.Tensor
        A tensor,  magnitude of a complex spectrogram,
        output from the spectral_magnitude function.
    hop_len : int
       Hop Length between frames in ms.
    tau : int
       Time constant in ms.

    Example
    -------
    >>> a = torch.Tensor([[3, 200]])
    >>> smooth_magnitude(a, tau=125)
    tensor([3, 25])
    """

    # compute time constant
    alpha = math.exp(-hop_len / tau)

    # number of frames
    n_frames = mag.shape[1]

    # recursive averaging
    mag_smoothed = torch.zeros_like(mag)
    for idx in range(n_frames):
        mag_smoothed[:, idx, :] = alpha * mag_smoothed[:, idx, :] + (1 - alpha) * mag[:, idx, :]

    # store a frame every 125 ms, i.e. every 10th frame (for window length of 25ms)    
    mag_smoothed = mag_smoothed[:, ::10, :]

    return mag_smoothed


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

mag_smoothed = smooth_magnitude(mag)
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
