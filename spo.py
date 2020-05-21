import numpy as np
import time
from scipy import signal
import utils


def low_envelope(s, dmin=1):
    lmin = (np.diff(np.sign(np.diff(s))) > 0).nonzero()[0] + 1
    lmin = lmin[[i+np.argmin(s[lmin[i:i+dmin]]) for i in range(0, len(lmin), dmin)]]
    return lmin


def smooth(a, WSZ):
    # a: NumPy 1-D array containing the data to be smoothed
    # WSZ: smoothing window size needs, which must be odd number,
    # as in the original MATLAB implementation
    out0 = np.convolve(a, np.ones(WSZ, dtype=int), 'valid')/WSZ
    r = np.arange(1, WSZ-1, 2)
    start = np.cumsum(a[:WSZ-1])[::2]/r
    stop = (np.cumsum(a[:-WSZ:-1])[::2]/r)[::-1]
    return np.concatenate((start, out0, stop))


class Process(object):
    def __init__(self):
        self.frame_in = np.zeros((20, 20, 3), np.uint8)
        self.buffer_size = 50
        self.times = []
        self.green_buffer = []
        self.red_buffer = []
        self.blue_buffer = []
        self.fps = 0
        self.t0 = time.time()
        self.spo = 0



    def update(self, roi):
        r = utils.extract_color(roi, 2)
        b = utils.extract_color(roi, 0)
        buffer_len = len(self.red_buffer)
        self.times.append(time.time() - self.t0)
        self.red_buffer.append(r)
        self.blue_buffer.append(b)

        if buffer_len == self.buffer_size:
            self.fps = float(buffer_len) / (self.times[-1] - self.times[0])
            band = [0.04, 0.045, 0.4, 0.45]  # Desired pass band, Hz
            red = self.get_filtered(self.red_buffer, band)
            band = [0.04, 0.045, 0.4, 0.45]  # Desired pass band, Hz

            blue = self.get_filtered(self.blue_buffer, band)
            self.get_oximeter(red, blue)

    def get_filtered(self, array, band):
        processed = np.array(array)
        processed = signal.detrend(processed)
        processed = utils.butter_bandpass_filter(processed, 0.3, 2.5, self.fps, order=3)
        signal_line = processed - smooth(processed, 5)
        signal_line = utils.butter_bandpass_filter(signal_line, 0.05, 0.15, self.fps, order=9)
        analytic_signal = signal.hilbert(signal_line)
        high = np.mean(np.abs(analytic_signal))
        low = np.mean(low_envelope(signal_line))
        d_signal = signal.detrend(signal_line)
        removed = filter(lambda x: low < x < high, d_signal)


        edges = [0, band[0], band[1], band[2], band[3], 0.5*self.fps]
        taps = signal.remez(3, edges, [0, 1, 0], Hz=self.fps, grid_density=20)
        filttered_color = np.convolve(taps, removed)[len(taps) // 2:]

        return filttered_color

    def get_oximeter(self, red, blue):
        fft_red = np.fft.fft(red)
        red_peaks, _ = signal.find_peaks(fft_red[1:])
        ratio_red = np.abs(np.array([fft_red[i] for i in red_peaks]))/np.abs(fft_red[0])

        fft_blue = np.fft.fft(blue)
        blue_peaks, _ = signal.find_peaks(fft_red[1:])
        ratio_blue = np.abs(np.array([fft_blue[i] for i in red_peaks]))/np.abs(fft_blue[0])
        min_len = min(len(ratio_red), len(ratio_blue))
        relation = np.divide(ratio_red[:min_len], ratio_blue[:min_len])
        o2_raw = 96.58 - -0.015*relation*100
        removed = filter(lambda x: x < 100, o2_raw)
        return removed, np.mean(removed).real()

    def reset(self):

        self.frame_in = np.zeros((20, 20, 3), np.uint8)
        self.buffer_size = 500
        self.times = []
        self.green_buffer = []
        self.red_buffer = []
        self.blue_buffer = []
        self.fps = 0
        self.t0 = time.time()
        self.spo = 0

    def get_spo(self):
        return self.spo
