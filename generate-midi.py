from mido import MidiFile, MidiTrack, Message
from music21 import *
import pandas as pd
import numpy as np
import math
import tensorflow as tf


def recreate_midi(df_first_notes, speed=20000):
    # function to take a dataframe created by something like parse_notes() or a gan and return a midi

    # Can start by reverse scaling the note:
    df_reversed = df_first_notes.copy()
    df_reversed['note'] = round(df_reversed['note'] * 88 + 20)
    df_reversed['duration'] /= 10
    df_reversed['time_since_last'] /= 10
    df_reversed.note = df_reversed.note.astype(int)
    df_reversed['velocity'] = 60  # create a uniform middling velocity

    # recreate the absolute time index and drop time_since_last (we'll recreate it with the stop signals)
    df_reversed['time_index'] = df_reversed.time_since_last.cumsum()
    df_reversed = df_reversed.drop(columns = 'time_since_last')

    # create a stop signal for each note at the appropriate time_index:
    for i in range(len(df_reversed)):
        stop_note = pd.DataFrame([[df_reversed.note[i], 0, 0, df_reversed.duration[i] + df_reversed.time_index[i]]],
                                 columns=['note', 'duration', 'velocity', 'time_index'])
        df_reversed = df_reversed.append(stop_note, ignore_index=True)
    df_reversed = df_reversed.sort_values('time_index').reset_index(drop=True)

    # recreate time_since last with the stop note signals
    df_reversed['time'] = [0] + [df_reversed.time_index[i+1] - df_reversed.time_index[i]
                                 for i in range(len(df_reversed)-1)]
    # and now we don't need duration or time_index so can drop those
    df_reversed = df_reversed.drop(columns = {'time_index','duration'})

    # finally, we need to scale the time since last note appropriately:
    df_reversed['time'] = round(df_reversed['time'] * speed)
    df_reversed.time = df_reversed.time.astype(int)

    # finally, recreate the midi and return
    mid_remade = MidiFile()
    track = MidiTrack()
    mid_remade.tracks.append(track)
    track.append(Message('program_change', program=0, time=0))
    for i in range(len(df_reversed)):
        track.append(Message('note_on', note=df_reversed.note[i], velocity=df_reversed.velocity[i], time=df_reversed.time[i]))

    return mid_remade

SEED_SIZE = 100

reconstructed_generator = tf.keras.models.load_model("start_2d_midi_generator")

noise = tf.random.normal([1, SEED_SIZE])
generated_notes = reconstructed_generator(noise, training=False).numpy().reshape((256,3))
midi_df = pd.DataFrame(generated_notes,columns=["note", "duration",'time_since_last'])
midi_remade = recreate_midi(midi_df)
midi_remade.save('generated_midi.mid')
