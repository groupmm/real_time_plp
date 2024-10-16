"""
Module Name: realtimeplp.py
Author:      Peter Meier
Email:       peter.meier@audiolabs-erlangen.de
Date:        2024-10-01
Version:     0.0.1
Description: Single File Real-Time PLP Implementation.
License:     MIT License (https://opensource.org/licenses/MIT)
"""

from dataclasses import dataclass, field
from typing import Any

import librosa
import numpy as np
from numpy._globals import _NoValue
from scipy import signal


@dataclass
class Peaks:
    """Represents detected peaks with frame indices and time positions."""

    i: np.ndarray  # frame indices
    t: np.ndarray  # time positions

    @classmethod
    def pick(cls, signal_x: np.ndarray, signal_t: np.ndarray, prominence: float = 0.01):
        """Detects peaks in a signal based on prominence."""
        # frame indices, where peaks appear
        peaks_i, _ = signal.find_peaks(signal_x, prominence=prominence)
        # time positions, where peaks appear
        peaks_t = signal_t[peaks_i]
        # Peaks object
        return cls(peaks_i, peaks_t)


@dataclass
class BeatActivation:
    """Beat Activation (Spectral Flux) from Audio Input"""

    N: int = 1024  # window size
    H: int = 512  # hop size
    samplerate: int = 0
    gamma: int = 1000  # parameter for logarithmic compression
    M: int = 10  # size of half centric local average window (no future values)

    _window_buffer: np.ndarray = field(init=False, repr=False)  # of size N samples
    _la_buffer: np.ndarray = field(init=False, repr=False)  # of size M frames
    _Y_last: np.ndarray = field(default=_NoValue, init=False, repr=False)

    activation_frame: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self._window_buffer = np.zeros(self.N)
        self._la_buffer = np.zeros(self.M)
        self.activation_frame = np.array([])

    def process(self, audio_frame: np.ndarray) -> np.ndarray:
        """Process input frame to output frame."""
        # STFT
        X_frame = self.stft(audio_frame)
        # Spectrogram
        Y_frame = self.spectrogram(X_frame)
        # Novelty
        Y_diff = np.diff(Y_frame, n=1, prepend=self._Y_last)  # Discrete Derivative
        Y_diff[Y_diff < 0] = 0  # Half-Wave Rectification
        nov = np.sum(Y_diff, axis=0)  # Accumulation
        # store last spectrogram Y_frame
        self._Y_last = Y_frame
        # local average
        la = self.local_average(nov)
        # normalization
        nov_norm = self.normalization(nov, la)
        # store current activation frame
        if nov_norm.size == 0:
            self.activation_frame = np.array([0.0])
        else:
            self.activation_frame = nov_norm
        return nov_norm

    def stft(self, audio_frame):
        """Process audio frame and compute STFT."""
        # roll frame
        self.roll_window_buffer(audio_frame)
        # compute stft
        X_frame = librosa.stft(
            self._window_buffer,
            n_fft=self.N,
            hop_length=self.H,
            win_length=self.N,
            window="hann",
            center=False,
            pad_mode="constant",
        )
        return X_frame

    def spectrogram(self, X_frame):
        """Compute online spectrogram."""
        Y_frame = np.log(1 + self.gamma * np.abs(X_frame))
        return Y_frame

    def roll_window_buffer(self, audio_frame):
        """Roll window_buffer with new audio_frame input."""
        self._window_buffer = np.roll(self._window_buffer, -self.H)
        self._window_buffer[-self.H :] = audio_frame

    def local_average(self, nov):
        """Compute the local average."""
        if len(nov) > 0:  # the first round, nov is empty [] --> no buffer roll!
            self._la_buffer = np.roll(self._la_buffer, -1)
            self._la_buffer[-1] = nov[0]
        # NOTE: LOCAL AVERAGE only looks back in time
        # update local average window buffer
        la = (1 / (self.M)) * np.sum(self._la_buffer)
        return la

    def normalization(self, nov, la):
        """Normalized novelty curve. Post-processing step."""
        nov_norm = nov - la  # 1. Subtracting Local Average
        nov_norm[nov_norm < 0] = 0  # 2. Half-Wave Rectification
        return nov_norm


@dataclass
class Tempogram:
    """Tempogram from Beat Activation Input"""

    N_time: int = 6  # window size given as time in seconds
    framerate: float = 0
    Theta: np.ndarray = field(default_factory=lambda: np.arange(60, 181, 1))
    H: int = field(default=1, repr=False)  # hop size (default=1 computing every frame)
    winfunc: Any = field(default_factory=lambda: np.hanning, repr=False)

    _tempo_buffer: np.ndarray = field(init=False, repr=False)

    tempogram_frame: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # NOTE: _tempo_buffer could be only half the size, like described in TISMIR.
        #       See half_window_method where we only roll the left half of the buffer.
        self._tempo_buffer = np.zeros(self.N)
        assert (
            self.framerate != 1
        ), f"Tempogram samplerate is not set: fs = {self.framerate}"
        self.tempogram_frame = np.array([])

    @property
    def N(self):
        """Get N in samples from N_time."""
        return round(self.N_time * self.framerate)

    @property
    def win(self):
        """Setup windows function with length N."""
        return self.winfunc(self.N)

    def process(self, activation_frame: np.ndarray) -> np.ndarray:
        """Process Input Frame to Output Frame."""
        self.half_window_method(activation_frame)
        L = self._tempo_buffer.shape[0]
        m = np.arange(L) / self.framerate
        K = len(self.Theta)
        X = np.zeros((K, 1), dtype=np.complex128)
        win = self.win
        for k in range(K):
            omega = self.Theta[k] / 60
            exponential = np.exp(-2 * np.pi * 1j * omega * m)
            X[k, 0] = np.sum(self._tempo_buffer * win * exponential)
        # store latest tempogram frame
        self.tempogram_frame = X
        return X

    def half_window_method(self, activation_frame):
        """Half window method: Only roll the left half of the buffer. Future is zero."""
        # the first round, frame is empty [] --> no buffer roll!
        if not len(activation_frame) == 0:
            nov_half = self._tempo_buffer[: self.N // 2]
            nov_half = np.roll(nov_half, -self.H)
            nov_half[-self.H :] = activation_frame
            self._tempo_buffer[: len(nov_half)] = nov_half


@dataclass
class Kernel:
    """Kernel class."""

    n: int
    k: Any
    tempo: int
    omega: float
    c: Any
    phase: float
    t_start: int
    t_end: int
    t: np.ndarray
    x: np.ndarray

    @classmethod
    def from_plp(
        cls,
        N: int,
        H: int,
        win: np.ndarray,
        Theta: np.ndarray,
        framerate: float,
        X: np.ndarray,
        n: int,
        tempogram: np.ndarray,
    ):
        """Init Kernel from PLP arguments."""
        k = np.argmax(tempogram[:, n])
        tempo = Theta[k]
        omega = (tempo / 60) / framerate
        c = X[k, n]
        phase = -np.angle(c) / (2 * np.pi)
        t_start = n * H
        t_end = t_start + N
        t = np.arange(t_start, t_end)
        x = win * np.cos(2 * np.pi * (t * omega - phase))
        return cls(n, k, tempo, omega, c, phase, t_start, t_end, t, x)


@dataclass
class PredominantLocalPulse:
    """Real-Time Predominant Local Pulse."""

    N_time: int = 6
    framerate: float = 0
    Theta: np.ndarray = field(default_factory=lambda: np.arange(60, 181, 1))
    lookahead: int = 0
    H: int = field(default=1, repr=False)

    stability: float = field(default=0, init=False, repr=False)
    current_tempo: float = field(default=0, init=False, repr=False)
    _pulse_buffer: np.ndarray = field(init=False, repr=False)
    _t: np.ndarray = field(init=False, repr=False)
    _cursor: int = field(default=0, init=False, repr=False)  # buffer read position
    _last_beat_distance: int = field(default=1_000_000_000, init=False, repr=False)
    _max_peak_amplitude: float = field(default=-1, init=False, repr=False)

    current_kernel: Kernel = field(init=False, repr=False)

    def __post_init__(self):
        self._pulse_buffer = np.zeros(self.N)  # plp buffer
        self._t = np.arange(self.N) / self.framerate  # time array for plp buffer
        self._cursor = (self.N // 2) + self.lookahead

    @property
    def cursor(self):
        """Get buffer read position considering lookahead."""
        return self._cursor

    @property
    def N(self):
        """Get N in samples."""
        return round(self.N_time * self.framerate)

    @property
    def win(self):
        """Get window function with length N."""
        return np.hanning(self.N) if self.N else None

    @property
    def current_buffer(self):
        """Get the current plp buffer (_pulse_buffer)."""
        return self._pulse_buffer

    def process(self, tempogram_frame: np.ndarray) -> bool:
        """Process Input Frame to Output Frame."""
        # ROLL BUFFER with new block; set new values to zero
        self._pulse_buffer = np.roll(self._pulse_buffer, -self.H)
        self._pulse_buffer[-self.H :] = 0
        # Tempogram
        tempogram = np.abs(tempogram_frame)
        # KERNEL COMPUTATION
        kernel = Kernel.from_plp(
            N=self.N,
            H=self.H,
            win=self.win,
            Theta=self.Theta,
            framerate=self.framerate,
            X=tempogram_frame,
            n=0,
            tempogram=tempogram,
        )  # n=0: Arrays have only 1 column
        # Overlapp-Add new kernel to buffer
        self._pulse_buffer = self._pulse_buffer + kernel.x
        # Set current tempo
        self.current_tempo = kernel.tempo
        # Store current kernel
        self.current_kernel = kernel
        return self._pulse_buffer

    def detect_beat(self, pulse_buffer: np.ndarray) -> np.ndarray:
        """Check pulse_buffer for peak at current time position (= half_buffer_index)"""
        buffer_peaks = Peaks.pick(pulse_buffer, self._t)
        future_beats = buffer_peaks.i[buffer_peaks.i > self._cursor]
        if len(future_beats) > 0:  # if there are future beats
            closest_future_beat = future_beats[0]
        else:  # if there are no future beats: avoid IndexError if future_beats is empty
            closest_future_beat = 0
        distance_to_closest_future_beat = closest_future_beat - self._cursor
        # if distance got bigger = beat detected
        beat_detected = distance_to_closest_future_beat > self._last_beat_distance
        self._last_beat_distance = distance_to_closest_future_beat
        if beat_detected:
            # Analyze amplitude of current beat position
            peak_amplitude = pulse_buffer[self._cursor]
            if peak_amplitude > self._max_peak_amplitude:
                self._max_peak_amplitude = peak_amplitude
            # Update stability value for current beat position
            self.stability = peak_amplitude / self._max_peak_amplitude
        return beat_detected


@dataclass
class ControlSignals:
    """Beat-Synchronous Control Signals Based on Real-Time PLP."""

    plp: PredominantLocalPulse

    def __post_init__(self):
        self._max_window_sum = self.calc_max_window_sum()
        self._alpha = self.calc_alpha()

    @property
    def normalized_buffer(self):
        """Get normalized current pulse buffer with values in [-1, 1]."""
        return self.normalize_buffer(self.plp.current_buffer)

    @property
    def alpha_lfo(self):
        """Get alpha-LFO value."""
        alpha_plp = self.calc_alpha_plp(self.plp.current_buffer)
        return alpha_plp[self.plp.cursor]

    @property
    def gamma_lfo(self):
        """Get gamma-LFO value."""
        gamma_plp = self.calc_gamma_plp(self.plp.current_buffer)
        return gamma_plp[self.plp.cursor]

    @property
    def beta_confidence(self):
        """Get beta-confidence value."""
        beta_envelope = self.calc_beta_envelope(self.plp.current_buffer)
        return beta_envelope[self.plp.cursor]

    @property
    def gamma_confidence(self):
        """Get gamma-confidence value."""
        gamma_envelope = self.calc_gamma_envelope(self.plp.current_buffer)
        return gamma_envelope[self.plp.cursor]

    @property
    def alpha_plp(self):
        """Get alpha-normalized PLP for current pulse buffer.
        See \alpha_{n_0} from DAFx paper 2024.
        """
        return self.calc_alpha_plp(self.plp.current_buffer)

    @property
    def gamma_plp(self):
        """Get gamma-normalized PLP for current pulse buffer."""
        return self.calc_gamma_plp(self.plp.current_buffer)

    @property
    def beta_envelope(self):
        """Calculate beta envelope for current pulse buffer."""
        return self.calc_beta_envelope(self.plp.current_buffer)

    @property
    def gamma_envelope(self):
        """Calculate gamma envelope."""
        return self.calc_gamma_envelope(self.plp.current_buffer)

    def overlap_add_kernel_windows(self, window: np.ndarray, H: int, L: int):
        """Overlap-add of kernel windows
        window: Window to overlap add.
        H: Hopsize to shift the window for overlapping.
        L: Total length of the overlap-added window sum (e.g. two times window length).
        """
        N = len(window)  # The number of samples in the window.
        M = np.floor((L - N) / H).astype(int) + 1
        w_sum = np.zeros(L)
        for m in range(M):
            w_shifted = np.zeros(L)
            w_shifted[m * H : m * H + N] = window
            w_sum = w_sum + w_shifted
        return w_sum

    def calc_max_window_sum(self):
        """Calculate max value of window sum for normalization purposes.
        See constant C from DAFx paper 2024.
        """
        window = signal.get_window(window="hann", Nx=self.plp.N)
        return np.sum(window).astype(int)

    def normalize_buffer(self, buffer):
        """Normalize pulse buffer with _max_window_sum for values in [-1, 1]."""
        return buffer / self._max_window_sum

    def calc_alpha(self):
        """Calculate alpha normalization for pulse buffer.
        See \alpha_{n_0} from DAFx paper 2024.
        """
        H = 1
        L = 2 * self.plp.N
        window = signal.get_window(window="hann", Nx=self.plp.N)
        window_sum = self.overlap_add_kernel_windows(window, H, L)
        alpha = window_sum[(len(window_sum) // 2) :]
        alpha = self.normalize_buffer(alpha)
        return alpha

    def calc_alpha_plp(self, buffer):
        """Calculate alpha-normalized PLP."""
        norm_buffer = self.normalize_buffer(buffer)
        alpha_plp = norm_buffer / self._alpha
        return alpha_plp

    def calc_gamma_plp(self, buffer):
        """Calculate gamma-normalized PLP."""
        alpha_plp = self.calc_alpha_plp(buffer)
        beta_envelope = self.calc_beta_envelope(buffer)
        gamma_plp = alpha_plp / beta_envelope
        return gamma_plp

    def calc_beta_envelope(self, buffer):
        """Calculate beta envelope."""
        alpha_plp = self.calc_alpha_plp(buffer)
        beta_envelope = np.abs(signal.hilbert(alpha_plp))
        return beta_envelope

    def calc_gamma_envelope(self, buffer):
        """Calculate gamma envelope."""
        beta_envelope = self.calc_beta_envelope(buffer)
        gamma_envelope = self._alpha * beta_envelope
        return gamma_envelope

    def calc_gamma_envelope2(self, buffer):
        """Calculate gamma envelope directly from the buffer envelope.
        Produces more noisy signal on the edges. Better use calc_gamma_envelope().
        """
        norm_buffer = self.normalize_buffer(buffer)
        gamma_envelope = np.abs(signal.hilbert(norm_buffer))
        return gamma_envelope


@dataclass
class RealTimeBeatTracker:
    """Real-Time Beat Tracker with Predominant Local Pulse (PLP)"""

    activation: BeatActivation
    tempogram: Tempogram
    plp: PredominantLocalPulse
    cs: ControlSignals

    def process(self, audio_frame: np.ndarray) -> bool:
        """Run Beat Tracker Frame for Frame."""

        activation_frame = self.activation.process(audio_frame)
        tempogram_frame = self.tempogram.process(activation_frame)
        pulse_buffer = self.plp.process(tempogram_frame)
        beat_detected = self.plp.detect_beat(pulse_buffer)
        return beat_detected

    @classmethod
    def from_args(
        cls,
        N=1024,
        H=512,
        samplerate=48000,
        N_time=6,
        Theta=np.arange(60, 181, 1),
        lookahead=0,
    ):
        """Create Beat Tracker for Real-Time"""
        act = BeatActivation(N=N, H=H, samplerate=samplerate)
        tempo = Tempogram(N_time=N_time, framerate=(samplerate / H), Theta=Theta)
        pulse = PredominantLocalPulse(
            N_time=N_time, framerate=(samplerate / H), Theta=Theta, lookahead=lookahead
        )
        cs = ControlSignals(plp=pulse)
        return cls(activation=act, tempogram=tempo, plp=pulse, cs=cs)


@dataclass
class BeatAnalyzer:
    """Beat analyzer class wrapper for storing all beat tracker frames over time."""

    tracker: RealTimeBeatTracker
    frame_indices: list = field(default_factory=list, init=False)
    frame_times: list = field(default_factory=list, init=False)
    audio_frames: list = field(default_factory=list, init=False)
    activation_frames: list = field(default_factory=list, init=False)
    tempogram_frames: list = field(default_factory=list, init=False)
    tempo_values: list = field(default_factory=list, init=False)
    kernels: list = field(default_factory=list, init=False)
    plp_buffers: list = field(default_factory=list, init=False)

    alpha_lfos: list = field(default_factory=list, init=False)
    gamma_lfos: list = field(default_factory=list, init=False)
    beta_confs: list = field(default_factory=list, init=False)
    gamma_confs: list = field(default_factory=list, init=False)

    alpha_plps: list = field(default_factory=list, init=False)
    gamma_plps: list = field(default_factory=list, init=False)
    beta_envs: list = field(default_factory=list, init=False)
    gamma_envs: list = field(default_factory=list, init=False)

    beat_detection_frames: list = field(default_factory=list, init=False)

    _frame_count: int = field(default=0, init=False)

    def process(self, audio_frame: np.ndarray) -> bool:
        """Run beat tracker frame by frame and store frame-based data for analyzing.
        Wrapper function for RealTimeBeatTracker.process()."""

        # process current audio frame
        beat_detected = self.tracker.process(audio_frame)
        # store frame-based data
        self.frame_indices.append(self._frame_count)
        self.frame_times.append(self.get_frame_time(self._frame_count))
        self.audio_frames.append(audio_frame)
        self.activation_frames.append(self.tracker.activation.activation_frame)
        self.tempogram_frames.append(self.tracker.tempogram.tempogram_frame)
        self.tempo_values.append(self.tracker.plp.current_tempo)
        self.kernels.append(self.tracker.plp.current_kernel)
        self.plp_buffers.append(self.tracker.cs.normalized_buffer)
        self.beat_detection_frames.append(beat_detected)

        self.alpha_lfos.append(self.tracker.cs.alpha_lfo)
        self.gamma_lfos.append(self.tracker.cs.gamma_lfo)
        self.beta_confs.append(self.tracker.cs.beta_confidence)
        self.gamma_confs.append(self.tracker.cs.gamma_confidence)

        self.alpha_plps.append(self.tracker.cs.alpha_plp)
        self.gamma_plps.append(self.tracker.cs.gamma_plp)
        self.beta_envs.append(self.tracker.cs.beta_envelope)
        self.gamma_envs.append(self.tracker.cs.gamma_envelope)

        # set new frame count
        self._frame_count += 1
        return beat_detected

    def get_frame_time(self, frame_count: int) -> float:
        """Get frame time from frame count based on framerate."""
        return frame_count / self.tracker.plp.framerate


if __name__ == "__main__":
    tracker = RealTimeBeatTracker.from_args()
    analyzer = BeatAnalyzer(tracker)
    print(tracker)
    print(analyzer)
