import os

import librosa
import soundfile as sf
import numpy as np
from chord import *

'''
Data_Augmenter
augment(audio_file,chord_file)
augment_audio(audio_file)
augment_chords_file(chord_file)
'''

interval_dict = {
    0: 'P1',
    1: 'm2',
    2: 'M2',
    3: 'm3',
    4: 'M3',
    5: 'P4',
    6: 'A4',  # or 'd5' for diminished fifth
    7: 'P5',
    8: 'm6',
    9: 'M6',
    10: 'm7',
    11: 'M7',
    12: 'P8',
}

# Define the output directories relative to the current working directory
audio_output_dir = 'audiodata/'
chord_output_dir = 'chorddata/'

# Ensure that output directories exist
os.makedirs(audio_output_dir, exist_ok=True)
os.makedirs(chord_output_dir, exist_ok=True)

# Audio Data Augmentation: Stretching And Pitch Shifting
def pitch_shift_and_save(y, sr, n_steps, output_file):
    """Shift the pitch of an audio signal and save it to a stereo file."""
    # Shift pitch for both channels
    y_shifted_left = librosa.effects.pitch_shift(y[0], sr=sr, n_steps=n_steps)
    y_shifted_right = librosa.effects.pitch_shift(y[1], sr=sr, n_steps=n_steps)
    # Combine both channels
    y_shifted_stereo = np.vstack([y_shifted_left, y_shifted_right])
    # Write the stereo file
    sf.write(output_file, y_shifted_stereo.T, sr, format='WAV')  # Transpose array to shape (n_frames, n_channels)


def augment_audio(file_path):
    y, sr = librosa.load(file_path, sr=None, mono=False)  # Load in stereo
    if y.ndim == 1:  # If the loaded file is mono, make it stereo
        y = np.tile(y, (2, 1))
    speeds = [0.75, 1.0, 1.25]  # Speed factors
    #speeds = [1.0]  # Speed factors
    semitone_shifts = list(range(-6, 6))  # -6 semitones to +5 semitones

    for speed in speeds:
        # Adjust the speed for both channels
        y_changed_speed_left = librosa.effects.time_stretch(y[0], rate=speed)
        y_changed_speed_right = librosa.effects.time_stretch(y[1], rate=speed)
        y_changed_speed = np.vstack([y_changed_speed_left, y_changed_speed_right])

        output_speed = int(100 * speed)
        file_name = file_path.split('.')[0]  # Assuming file has an extension

        for shift in semitone_shifts:
            output_file = os.path.join(audio_output_dir,
                                       f"{os.path.basename(file_name)}_speed{output_speed}_pitch{shift}.wav")
            pitch_shift_and_save(y_changed_speed, sr, shift, output_file)
            print(f"Saved: {output_file}")

def augment_chord_line(line, time_stretch_factor, semitones):
    parts = line.split()
    start_time, end_time, chord_str = float(parts[0]), float(parts[1]), parts[2]

    # Apply time stretching
    start_time *= time_stretch_factor
    end_time *= time_stretch_factor

    # Parse the chord and transpose it
    chord = parse_chord_string(chord_str)
    chord.transpose(interval_dict[abs(semitones)], 1 if semitones >= 0 else -1)

    # Format the augmented line
    augmented_line = f"{start_time:.2f} {end_time:.2f} {chord}"
    return augmented_line

def augment_chords_file(input_file_path):
    with open(input_file_path, 'r') as input_file:
        lines = input_file.readlines()

    base_name = os.path.splitext(input_file_path)[0]

    # Time-stretch factors for different speeds
    speed_factors = {
        0.75: 1.0 / 0.75,
        1.0: 1.0,
        1.25: 1.0 / 1.25
    }
    semitone_shifts = list(range(-6, 6))  # -6 semitones to +5 semitones

    for speed, time_stretch_factor in speed_factors.items():
        output_speed = int(100 * speed)
        for shift in semitone_shifts:
            output_file_name = os.path.join(chord_output_dir,
                                            f"{os.path.basename(base_name)}_speed{output_speed}_pitch{shift}.txt")

            with open(output_file_name, 'w') as output_file:
                for line in lines:
                    line = line.strip()  # Remove any whitespace at the start/end
                    if not line:  # Skip empty lines
                        continue
                    augmented_line = augment_chord_line(line, time_stretch_factor, shift)
                    output_file.write(augmented_line + '\n')
            print(f"Saved: {output_file_name}")



def augment(audio_file,chord_file):
    augment_audio(audio_file)
    augment_chords_file(chord_file)
