"""
Module Name: beatcli.py
Author:      Peter Meier
Email:       peter.meier@audiolabs-erlangen.de
Date:        2024-10-01
Version:     0.0.1
Description: A Real-Time Beat Tracking Dashboard for the Terminal.
License:     MIT License (https://opensource.org/licenses/MIT)
"""

import argparse
import datetime
import threading

import numpy as np
import sounddevice as sd
from pythonosc import udp_client

from realtimeplp import RealTimeBeatTracker

# TODO: Add --learn mode for mapping OSC signals.
# Output all OSC channels in a row with 1 second delay between.

# Argparse
parser = argparse.ArgumentParser(description="Beat (C)ommand (L)ine (I)nterface.")
parser.add_argument(
    "-l",
    "--list-devices",
    action="store_true",
    help="show list of audio devices and exit",
)
parser.add_argument(
    "--device",
    metavar="ID",
    type=int,
    help="(%(default)s) device id for sounddevice input",
)
parser.add_argument(
    "--channel",
    default=1,
    metavar="NUMBER",
    type=int,
    help="(%(default)s) channel number for sounddevice input",
)
parser.add_argument(
    "--samplerate",
    default=44100,
    metavar="FS",
    type=int,
    help="(%(default)s) samplerate for sounddevice",
)
parser.add_argument(
    "--blocksize",
    default=512,
    metavar="SAMPLES",
    type=int,
    help="(%(default)s) blocksize for sounddevice",
)
parser.add_argument(
    "--tempo",
    nargs=2,
    metavar=("LOW", "HIGH"),
    default=[60, 180],
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
parser.add_argument(
    "--ip", default="0.0.0.0", type=str, help="(%(default)s) ip address for OSC client"
)
parser.add_argument(
    "--port", default=5005, type=int, help="(%(default)s) port for OSC client"
)
args = parser.parse_args()
# List devices
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
low, high = args.tempo
if high <= low:
    parser.error("HIGH must be greater than LOW")


# Sounddevice Settings
sd.default.blocksize = args.blocksize
sd.default.samplerate = args.samplerate
sd.default.latency = 0  # in seconds
sd.default.channels = args.channel  # number of channels (both in and out)
sd.default.device = (args.device, args.device)  # (in_device_id, out_device_id)
device = sd.query_devices(args.device, "input")

# OSC Client
client = udp_client.SimpleUDPClient(args.ip, args.port)

beat = RealTimeBeatTracker.from_args(
    N=2 * args.blocksize,
    H=args.blocksize,
    samplerate=args.samplerate,
    N_time=args.kernel,
    Theta=np.arange(low, high + 1, 1),
    lookahead=args.lookahead,
)

# Global Variables
stability = []
tempo = []


def callback(indata, _frames, _time, status):
    """Audio callback."""
    if status:
        print(status)
    # counting from 0: index of last channel = number of channels - 1
    beat_detected = beat.process(indata[:, args.channel - 1])

    # OSC messages on framerate level
    # shift value range from [-1,1] to [0,1] for DAW (like Ableton)
    client.send_message("/alpha-lfo", (beat.cs.alpha_lfo + 1) / 2)  # [-1,1] to [0,1]
    client.send_message("/gamma-lfo", (beat.cs.gamma_lfo + 1) / 2)  # [-1,1] to [0,1]
    client.send_message("/beta-conf", beat.cs.beta_confidence)
    client.send_message("/gamma-conf", beat.cs.gamma_confidence)

    # OSC / console output on beat level
    if beat_detected:
        # send OSC messages
        client.send_message("/stability", beat.plp.stability)
        client.send_message("/tempo", int(beat.plp.current_tempo))
        # print to console
        print(
            f"OSC to {args.ip}:{args.port}:",
            f'time={datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]}',
            f"tempo={beat.plp.current_tempo}",
            f"stability={beat.cs.beta_confidence:.3f}",
        )
        tempo.append(beat.plp.current_tempo)
        stability.append(beat.cs.beta_confidence)


if __name__ == "__main__":
    # Print Arguments
    print("Beat (C)ommand (L)ine (I)nterface:", vars(args))
    # Thread Event for keeping sounddevice running
    event = threading.Event()

    # Start Sounddevice Input Stream
    with sd.InputStream(
        callback=callback, samplerate=args.samplerate, finished_callback=event.set
    ):
        try:
            event.wait()
        except KeyboardInterrupt:
            print("")
            print("--- beat statistics ---")
            print(f"{len(tempo)} beats transmitted")
            print(
                f"tempo min/avg/max/stddev = "
                f"{min(tempo):.2f}/{np.mean(tempo):.2f}/"
                f"{max(tempo):.2f}/{np.std(tempo):.2f}"
            )
            print(
                f"stability min/avg/max/stddev = "
                f"{min(stability):.3f}/{np.mean(stability):.3f}/"
                f"{max(stability):.3f}/{np.std(stability):.3f}"
            )
