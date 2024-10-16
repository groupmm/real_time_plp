"""
Module Name: rockyourbeats.py
Author:      Peter Meier
Email:       peter.meier@audiolabs-erlangen.de
Date:        2024-10-01
Version:     0.0.1
Description: An Educational Beat Game Prototype Using Real-Time PLP.
License:     MIT License (https://opensource.org/licenses/MIT)
"""

import collections
import datetime
import sys
from statistics import mean
from typing import Deque

import numpy as np  # needed for sounddevice callback
import pygame
import sounddevice as sd

from realtimeplp import Peaks, RealTimeBeatTracker

assert np  # eliminate "imported but unused" lint error


# Sounddevice constants
SAMPLERATE = 44100
BUFFERSIZE = 512
LOW = 60
HIGH = 180
LOOKAHEAD = 0
KERNEL = 6  # Kernel size in seconds
SCENEMIN = 1.5  # Displaying a portion of the kernel window in the game
SCENEMAX = 4.5  # Displaying a portion of the kernel window in the game
LATENCY = 0
DEVICE = None
CHANNELS = 1


# Pygame constants
WIDTH = 960
HEIGHT = 640
FLOORHEIGHT = 128
FLOORLEVEL = HEIGHT - FLOORHEIGHT


# Pygame init
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rock Your Beats")
clock = pygame.time.Clock()
myfont = pygame.font.Font(pygame.font.match_font("Arial"), 38)


background = pygame.image.load("assets/graphics/background.png").convert_alpha()


rock = pygame.image.load("assets/graphics/rock.png").convert_alpha()
rock = pygame.transform.scale(rock, (64, 64))
rock_rect = rock.get_rect(midbottom=(WIDTH / 2, FLOORLEVEL - 150))
rock_velocity = 0


slime = pygame.image.load("assets/graphics/slime.png").convert_alpha()
slime = pygame.transform.scale(slime, (64, 64))
slime_hit = pygame.image.load("assets/graphics/slime_hit.png").convert_alpha()
slime_hit = pygame.transform.scale(slime_hit, (64, 64))


slime_hud = pygame.image.load("assets/graphics/slime_dead.png").convert_alpha()
slime_hud = pygame.transform.scale(slime_hud, (64, 64))
slime_hud_rect = slime_hud.get_rect(topleft=(32, -10))


x_hud = pygame.image.load("assets/graphics/hudX.png").convert_alpha()
x_hud = pygame.transform.scale(x_hud, (64, 64))

points = 0

# array of display numbers from PNG graphics
numbers = []
for x in range(10):
    number = pygame.image.load(f"assets/graphics/hud{x}.png").convert_alpha()
    number = pygame.transform.scale(number, (64, 64))
    numbers.append(number)

tracker = RealTimeBeatTracker.from_args(
    N=2 * BUFFERSIZE,
    H=BUFFERSIZE,
    samplerate=SAMPLERATE,
    N_time=KERNEL,
    Theta=np.arange(LOW, HIGH + 1, 1),
    lookahead=LOOKAHEAD,
)

stability_buffer: Deque = collections.deque([0], maxlen=5)

# track beats on screen to detect changes
beats_hit_indices: list = []
beats_len: int = 0


def callback(indata, frames, time, status):
    """Sounddevice callback function."""
    if status:
        print(status)
    beat_detected = tracker.process(indata[:, CHANNELS - 1])
    if beat_detected:
        # print to console
        print(
            f'time={datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]}',
            f"tempo={tracker.plp.current_tempo}",
            f"stability={tracker.cs.beta_confidence:.3f}",
        )
        stability_buffer.append(tracker.cs.beta_confidence)


pygame.joystick.init()
joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
for joy in joysticks:
    joy.init()

# sounddevice stream
with sd.InputStream(
    device=DEVICE,
    channels=CHANNELS,
    samplerate=SAMPLERATE,
    blocksize=BUFFERSIZE,
    latency=LATENCY,
    callback=callback,
):
    # pygame loop
    while True:

        t = np.arange(tracker.cs.alpha_plp.shape[0]) / tracker.plp.framerate
        buffer_peaks = Peaks.pick(tracker.cs.alpha_plp, t)

        peaks = buffer_peaks.t
        beats = [x for x in peaks if x < SCENEMAX and x > SCENEMIN]

        # if a beat leaves the scene
        if len(beats) < beats_len:
            # reduce all hit indices by 1
            beats_hit_indices = [x - 1 for x in beats_hit_indices]
            # remove negative hit indices
            beats_hit_indices = [x for x in beats_hit_indices if x >= 0]

        # update new beats len
        beats_len = len(beats)

        # create hit mask
        beats_hit_mask = [0 for _ in beats]
        if beats_hit_mask:  # avoid empty array
            for i in beats_hit_indices:
                beats_hit_mask[i] = 1

        beats_under_rock = [x for x in beats if x < 3.05 and x > 2.95]
        beats_mid_index = (len(beats) - 1) // 2

        # check event loop
        for event in pygame.event.get():
            # quit game window
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            # press spacebar
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    rock_rect.bottom = FLOORLEVEL
                    rock_velocity = 10
                    if beats_under_rock:
                        beats_hit_indices.append(beats_mid_index)
                        points += 1
            # press joystickbutton
            if event.type == pygame.JOYBUTTONDOWN:
                rock_rect.bottom = FLOORLEVEL
                rock_velocity = 10
                if beats_under_rock:
                    beats_hit_indices.append(beats_mid_index)
                    points += 1

        # graphics
        screen.blit(background, (0, 0))
        screen.blit(slime_hud, slime_hud_rect)
        screen.blit(x_hud, x_hud.get_rect(topleft=(96, 0)))

        # points display
        points_display = f"{points:003d}"  # int with leading zeros as str, e.g. "060"

        screen.blit(
            numbers[int(points_display[0])],
            numbers[int(points_display[0])].get_rect(topleft=(144, 0)),
        )
        screen.blit(
            numbers[int(points_display[1])],
            numbers[int(points_display[1])].get_rect(topleft=(176, 0)),
        )
        screen.blit(
            numbers[int(points_display[2])],
            numbers[int(points_display[2])].get_rect(topleft=(208, 0)),
        )

        # tempo display
        tempo = tracker.plp.current_tempo
        tempo_display = f"{tempo:003d}"  # int with leading zeros as str, e.g. "060"

        screen.blit(
            numbers[int(tempo_display[0])],
            numbers[int(tempo_display[0])].get_rect(topright=(WIDTH - 192, 0)),
        )
        screen.blit(
            numbers[int(tempo_display[1])],
            numbers[int(tempo_display[1])].get_rect(topright=(WIDTH - 160, 0)),
        )
        screen.blit(
            numbers[int(tempo_display[2])],
            numbers[int(tempo_display[2])].get_rect(topright=(WIDTH - 128, 0)),
        )

        bpm = myfont.render("BPM", True, (150, 150, 150))
        screen.blit(bpm, bpm.get_rect(topright=(WIDTH - 48, 8)))

        # stability threshold
        if mean(stability_buffer) > 0.5:
            # draw beat creatures
            for i, beat in enumerate(beats):
                pos = np.interp(beat, [SCENEMIN, SCENEMAX], [0, WIDTH])
                slime_rect = slime.get_rect(midbottom=(pos, FLOORLEVEL))
                if beats_hit_mask[i]:
                    screen.blit(slime_hit, slime_rect)
                else:
                    screen.blit(slime, slime_rect)

        screen.blit(rock, rock_rect)
        if rock_rect.bottom < FLOORLEVEL - 150:
            rock_velocity = 0
        rock_rect.y -= rock_velocity

        pygame.display.update()
        clock.tick(int(SAMPLERATE / BUFFERSIZE))  # frames per second
