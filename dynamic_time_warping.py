from __future__ import division
import matplotlib.pylab as P
import librosa
import soundfile as sf
import librosa.display
from scipy.interpolate import PchipInterpolator as mono_interp
from scipy.signal import decimate



def variable_phase_vocoder(D, times_steps, hop_length=None):
    n_fft = 2 * (D.shape[0] - 1)

    if hop_length is None:
        hop_length = int(n_fft // 4)

    # Create an empty output array
    d_stretch = P.zeros((D.shape[0], len(time_steps)), D.dtype, order='F')

    # Expected phase advance in each bin
    phi_advance = P.linspace(0, P.pi * hop_length, D.shape[0])

    # Phase accumulator; initialize to the first sample
    phase_acc = P.angle(D[:, 0])

    # Pad 0 columns to simplify boundary logic
    D = P.pad(D, [(0, 0), (0, 2)], mode='constant')

    for (t, step) in enumerate(time_steps):
        columns = D[:, int(step):int(step + 2)]

        # Weighting for linear magnitude interpolation
        alpha = P.mod(step, 1.0)
        mag = ((1.0 - alpha) * abs(columns[:, 0])
               + alpha * abs(columns[:, 1]))

        # Store to output array
        d_stretch[:, t] = mag * P.exp(1.j * phase_acc)

        # Compute phase advance
        dphase = (P.angle(columns[:, 1])
                  - P.angle(columns[:, 0])
                  - phi_advance)

        # Wrap to -pi:pi range
        dphase = dphase - 2.0 * P.pi * P.around(dphase / (2.0 * P.pi))

        # Accumulate phase
        phase_acc += phi_advance + dphase

    return d_stretch

def close_points(X, s=1):
  lambda_s = 1
  lambda_c = s
  X = P.array(X)
  K, N = X.shape
  M = P.zeros((N, N))
  M[range(N), range(N)] = 2*lambda_c/(N-1) + lambda_s/N
  d = P.diag(P.ones(N-1), 1)
  M = M - lambda_c*(d + d.T)/(N-1)
  M[0, 0] = lambda_s/N
  M[-1, -1] = lambda_s/N
  M[0, 1] = 0
  M[-1, -2] = 0
  Mi = P.pinv(M)
  smooth_X = (lambda_s/N)*Mi.dot(X.T).T
  return smooth_X

print("Loading Audio...")
y1, fs = sf.read("./Dataset/american/arctic_a0001.wav")
y2, fs = sf.read("./Dataset/indian/arctic_a0001.wav")
# Add some simple padding
i1 = P.argmax( y1 > P.sqrt((y1**2).mean())/3 )
i2 = P.argmax( y2 > P.sqrt((y2**2).mean())/3 )
I = max(i1, i2)*2
z1 = y1[i1//5:(i1//5)*2]
y1 = P.hstack([z1]*((I-i1)//len(z1)) + [z1[:((I - i1)%len(z1))]] + [y1])
z2 = y2[i2//5:(i2//5)*2]
y2 = P.hstack([z2]*((I-i2)//len(z2)) + [z2[:((I - i2)%len(z2))]] + [y2])
print("Setting padding to {0:.2f} s".format(I/fs))
# manually downsample by factor of 2
fs = fs//2
y1 = decimate(y1, 2, zero_phase=True)
y2 = decimate(y2, 2, zero_phase=True)
# Normalize loudness
v1 = P.sqrt((y1**2).mean())
v2 = P.sqrt((y2**2).mean())
y1 = y1/v1*.03
y2 = y2/v2*.03
print("Audio lengths (s)")
print(len(y1)/fs)
print(len(y2)/fs)

n_fft = 4410
hop_size = 2205

print("Starting DTW...")
y1_mfcc = librosa.feature.mfcc(y=y1, sr=fs, 
                              hop_length=hop_size, n_mfcc=80)
y2_mfcc = librosa.feature.mfcc(y=y2, sr=fs, 
                              hop_length=hop_size, n_mfcc=80)
D, wp = librosa.sequence.dtw(X=y1_mfcc, Y=y2_mfcc, metric='cosine')
print("Doing interpolation and warping...")
wp = wp[::-1, :]
y1_st, y1_end = wp[0, 0]*hop_size, wp[-1, 0]*hop_size
y2_st, y2_end = wp[0, 1]*hop_size, wp[-1, 1]*hop_size
y1 = y1[y1_st:y1_end]
y2 = y2[y2_st:y2_end]
wp[:, 0] = wp[:, 0] - wp[0,0]
wp[:, 1] = wp[:, 1] - wp[0,1]
wp_s = P.asarray(wp) * hop_size / fs
i, I = P.argsort(wp_s[-1, :])
x, y = close_points(
  P.array([wp_s[:,i]/wp_s[-1,i], wp_s[:,I]/wp_s[-1,I]]), s=1)
f = mono_interp(x, y, extrapolate=True)
yc,yo = (y1,y2) if i==1 else (y2, y1)
l_hop = 64
stft = librosa.stft(yc, n_fft=512, hop_length=l_hop)
z = len(yo)//l_hop + 1
t = P.arange(0, 1, 1/z)
time_steps = P.clip( f(t) * stft.shape[1], 0, None )
print("Beginning vocoder warping...")
warped_stft = variable_phase_vocoder(stft, time_steps, hop_length=l_hop)
y_warp = librosa.istft(warped_stft, hop_length=l_hop)

#
l = min(len(y_warp), len(y2))
y = y_warp[:l] + y1[:l]

print("Writing synced reading to file...")
sf.write("y1_warp.wav", y_warp[:l] * 10, fs)
sf.write("y1.wav", y1[:l], fs)
sf.write("synced.wav", y, fs)

