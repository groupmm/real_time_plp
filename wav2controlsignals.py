"""
Module Name: wav2controlsignal.py
Author:      Peter Meier
Email:       peter.meier@audiolabs-erlangen.de
Date:        2024-10-01
Version:     0.0.1
Description: Generate control signals output from wav audio input.
License:     MIT License (https://opensource.org/licenses/MIT)
"""

import argparse
import pathlib

import librosa
import numpy as np
import resampy
import soundfile as sf

from realtimeplp import BeatAnalyzer, RealTimeBeatTracker


def main():
    """Main function."""

    # Argparse
    parser = argparse.ArgumentParser(description="wav2controlsignal.py")
    parser.add_argument(
        "-i",
        dest="file",
        required=True,
        metavar="FILE",
        type=str,
        help="(%(default)s) input audio file",
    )
    parser.add_argument(
        "--buffersize",
        default=512,
        metavar="SAMPLES",
        type=int,
        help="(%(default)s) buffersize for framerate",
    )
    parser.add_argument(
        "--tempo",
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=[30, 240],
        type=int,
        help="(%(default)s) tempo range in BPM",
    )
    parser.add_argument(
        "--lookahead",
        default=0,
        metavar="FRAMES",
        type=int,
        help="(%(default)s) number of frames (samplerate / blocksize) to lookahead"
        " in time and get the next beat earlier to compensate for latency",
    )
    parser.add_argument(
        "--kernel",
        default=6,
        metavar="SIZE",
        type=int,
        help="(%(default)s) kernel size in seconds",
    )

    # parse input arguments
    args = parser.parse_args()
    low, high = args.tempo
    if high <= low:
        parser.error("HIGH must be greater than LOW")
    stem = pathlib.Path(args.file).stem
    suffix = pathlib.Path(args.file).suffix
    path = pathlib.Path(args.file).parents[0]  # filepath

    AUDIO = args.file  # "assets/audio/DrumBeat.wav"
    SR = librosa.get_samplerate(AUDIO)
    HOP = args.buffersize
    LOW = low
    HIGH = high
    LOOKAHEAD = args.lookahead
    KERNEL = args.kernel

    y, _ = librosa.load(AUDIO, sr=SR)
    audio_stream = librosa.stream(
        path=AUDIO, block_length=HOP, frame_length=1, hop_length=1, fill_value=0
    )
    tracker = RealTimeBeatTracker.from_args(
        N=2 * HOP,
        H=HOP,
        samplerate=SR,
        N_time=KERNEL,
        Theta=np.arange(LOW, HIGH + 1, 1),
        lookahead=LOOKAHEAD,
    )
    analyzer = BeatAnalyzer(tracker)
    # Process every frame of audio in the audio stream of the file
    for frame in audio_stream:
        analyzer.process(audio_frame=frame)

    # compute control signals
    alpha_lfo = analyzer.alpha_lfos
    gamma_lfo = analyzer.gamma_lfos
    beta_conf = analyzer.beta_confs
    gamma_conf = analyzer.gamma_confs

    # beat positions
    beats = np.zeros(len(y))
    for i, beat_detected in enumerate(analyzer.beat_detection_frames):
        if beat_detected:
            x = int(i * HOP)
            if x < len(beats):
                beats[x] = 1.0

    # tempo curve
    tempo = np.zeros(len(analyzer.frame_indices))
    low = analyzer.tracker.tempogram.Theta[0]
    high = analyzer.tracker.tempogram.Theta[-1]
    for i, kernel in enumerate(analyzer.kernels):
        tempo[i] = np.interp(kernel.tempo, [low, high], [-1, 1])

    # Resampling from feature framerate to audio samplerate
    alpha_lfo_resampled = resampy.resample(
        np.array(alpha_lfo), analyzer.tracker.plp.framerate, SR
    )
    gamma_lfo_resampled = resampy.resample(
        np.array(gamma_lfo), analyzer.tracker.plp.framerate, SR
    )
    beta_conf_resampled = resampy.resample(
        np.array(beta_conf), analyzer.tracker.plp.framerate, SR
    )
    gamma_conf_resampled = resampy.resample(
        np.array(gamma_conf), analyzer.tracker.plp.framerate, SR
    )
    tempo_resampled = resampy.resample(
        np.array(tempo), analyzer.tracker.plp.framerate, SR
    )

    alpha_lfo_resampled = alpha_lfo_resampled[: len(y)]
    gamma_lfo_resampled = gamma_lfo_resampled[: len(y)]
    beta_conf_resampled = beta_conf_resampled[: len(y)]
    gamma_conf_resampled = gamma_conf_resampled[: len(y)]
    beats = beats[: len(y)]
    tempo_resampled = tempo_resampled[: len(y)]

    # Write out wav files
    sf.write(
        pathlib.Path.joinpath(path, stem + "_alpha_lfo" + suffix).resolve(),
        alpha_lfo_resampled,
        int(SR),
    )
    sf.write(
        pathlib.Path.joinpath(path, stem + "_gamma_lfo" + suffix).resolve(),
        gamma_lfo_resampled,
        int(SR),
    )
    sf.write(
        pathlib.Path.joinpath(path, stem + "_beta_conf" + suffix).resolve(),
        beta_conf_resampled,
        int(SR),
    )
    sf.write(
        pathlib.Path.joinpath(path, stem + "_gamma_conf" + suffix).resolve(),
        gamma_conf_resampled,
        int(SR),
    )
    sf.write(
        pathlib.Path.joinpath(path, stem + "_beats" + suffix).resolve(), beats, int(SR)
    )
    sf.write(
        pathlib.Path.joinpath(path, stem + "_tempo" + suffix).resolve(),
        tempo_resampled,
        int(SR),
    )


if __name__ == "__main__":
    main()
