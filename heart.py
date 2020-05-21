import numpy as np
import time
from scipy import signal
import utils


class Process(object):
    def __init__(self):
        self.frame_in = np.zeros((20, 20, 3), np.uint8)

        self.samples = []
        self.buffer_size = 100
        self.times = []
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.frequency = []
        self.t0 = time.time()
        self.bpm = 0
        self.beats_per_minute = []
        self.peaks = []

    def update(self, roi):
        g = utils.extract_color(roi, 1)
        buffer_len = len(self.data_buffer)
        if abs(g - np.mean(self.data_buffer)) > 10 and buffer_len > 99:
            g = self.data_buffer[-1]

        self.times.append(time.time() - self.t0)
        self.data_buffer.append(g)
        if buffer_len > self.buffer_size:
            self.data_buffer = self.data_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            self.beats_per_minute = self.beats_per_minute[-self.buffer_size // 2:]
            buffer_len = self.buffer_size

        processed = np.array(self.data_buffer)

        if buffer_len == self.buffer_size:
            self.fps = float(buffer_len) / (self.times[-1] - self.times[0])
            even_times = np.linspace(self.times[0], self.times[-1], buffer_len)
            processed = signal.detrend(processed)
            interpolated = np.interp(even_times, self.times, processed)
            interpolated = np.hamming(buffer_len) * interpolated
            norm = interpolated / np.linalg.norm(interpolated)
            raw = np.fft.rfft(norm * 30)
            self.frequency = float(self.fps) / buffer_len * np.arange(buffer_len / 2 + 1)
            freqs = 60. * self.frequency
            self.fft = np.abs(raw) ** 2
            idx = np.where((freqs > 50) & (freqs < 180))
            pruned = self.fft[idx]
            pfreq = freqs[idx]
            self.frequency = pfreq
            self.fft = pruned
            idx2 = np.argmax(pruned)
            self.bpm = self.frequency[idx2]
            self.beats_per_minute.append(self.bpm)
            processed = utils.butter_bandpass_filter(processed, 0.8, 3, self.fps, order=3)
        self.samples = processed

    def reset(self):
        self.frame_in = np.zeros((20, 20, 3), np.uint8)
        self.samples = []
        self.times = []
        self.data_buffer = []
        self.fps = 0
        self.fft = []
        self.frequency = []
        self.t0 = time.time()
        self.bpm = 0
        self.beats_per_minute = []

    def get_bmp(self):
        # if self.beats_per_minute:
        #    return np.mean(self.beats_per_minute)
        #    else:
        #        return 0
        return self.bpm
