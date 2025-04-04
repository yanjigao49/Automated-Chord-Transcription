import csv

import librosa
import numpy as np
import os

from chord import parse_chord_string
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Parameters
audio_path = 'audiodata'  # Directory where audio files are stored
labels_path = 'chorddata'  # Directory where label files are stored
test_path = 'testdata' # Directory where test audio files are store
sample_rate = 44100  # Desired sample rate
chunk_duration = 1.0  # Duration of chunks in seconds
overlap = 0.8  # Overlap between chunks in seconds
n_fft = 2048  # Length of the FFT window
hop_length = 512  # Number of samples between successive frames
n_mels = 128  # Number of Mel bands to generate



# Function to load an audio file and its labels
def load_audio_with_labels(audio_file, label_file, sr):
    print(f"Loading audio file: {audio_file}")
    # Load the audio file
    audio, _ = librosa.load(audio_file, sr=sr)

    print(f"Loading label file: {label_file}")
    # Load the labels
    chord_data = []
    with open(label_file, 'r') as f:
        reader = csv.reader(f, delimiter=' ')
        for row in reader:
            start_time, end_time, chord_label = row
            chord_data.append((float(start_time), float(end_time), chord_label))
    print(f"Loaded {len(chord_data)} labels")

    return audio, chord_data


def segment_and_label(audio, chord_data, sr, chunk_duration, overlap):
    print("Segmenting audio and assigning labels...")
    chunk_size = int(chunk_duration * sr)
    step_size = int(chunk_size * (1 - overlap))

    # Extend chord_data with 'NC' for segments that are not covered by the provided labels
    extended_chord_data = fill_unlabeled_time_ranges(chord_data, audio_duration=len(audio) / sr)

    segments = []
    labels = []

    for start_idx in range(0, len(audio) - chunk_size + 1, step_size):
        # Time range of the current chunk
        start_time = start_idx / sr
        end_time = (start_idx + chunk_size) / sr

        # Find the label for the current chunk
        chord_label = find_label_for_chunk(start_time, end_time, extended_chord_data)

        # Use the enharmRespell function to ensure the chord label is within the valid notes
        chord = parse_chord_string(chord_label)
        chord.normalize()
        chord_label = chord.__str__()

        # Extract the chunk and store with the label
        chunk = audio[start_idx:start_idx + chunk_size]
        segments.append(chunk)
        labels.append(chord_label)
        print(f"Segment {len(segments)}: Start Time = {start_time}, End Time = {end_time}, Label = {chord_label}")

    print(f"Created {len(segments)} segments with labels")
    return segments, labels

def fill_unlabeled_time_ranges(chord_data, audio_duration):
    """
    Fill the audio duration with 'NC' for segments that do not have a chord label.
    """
    extended_chord_data = []
    current_time = 0.0

    for (start, end, chord) in chord_data:
        if start > current_time:
            # Fill the gap with 'NC'
            extended_chord_data.append((current_time, start, 'NC'))
            print(f"Added 'NC' label for gap: Start Time = {current_time}, End Time = {start}")
        extended_chord_data.append((start, end, chord))
        print(f"Label from data: Start Time = {start}, End Time = {end}, Label = {chord}")
        current_time = end

    # Fill the remaining time after the last chord label, if any
    if current_time < audio_duration:
        extended_chord_data.append((current_time, audio_duration, 'NC'))
        print(f"Added 'NC' label for trailing gap: Start Time = {current_time}, End Time = {audio_duration}")

    return extended_chord_data


def find_label_for_chunk(start_time, end_time, chord_data):
    """
    Find the label for a given chunk of audio based on the chord with the maximum duration within the chunk.
    """
    max_duration = 0
    selected_label = 'NC'  # Default label if no chord matches

    for (label_start, label_end, chord_label) in chord_data:
        # Calculate overlap with the segment
        overlap_start = max(start_time, label_start)
        overlap_end = min(end_time, label_end)
        overlap_duration = max(0, overlap_end - overlap_start)

        if overlap_duration > max_duration:
            max_duration = overlap_duration
            selected_label = chord_label

    return selected_label


# Function to extract Melspectrogram features
def extract_melspectrogram(audio_chunks, sr, n_fft, hop_length, n_mels):
    print("Extracting Melspectrograms...")
    melspectrograms = []
    for chunk in audio_chunks:
        # Extract the Melspectrogram for the audio chunk
        melspec = librosa.feature.melspectrogram(y=chunk, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        melspectrograms.append(melspec)
    print(f"Extracted {len(melspectrograms)} Melspectrograms")
    return melspectrograms


def process_file():
    print("Processing files...")
    audio_files = os.listdir(audio_path)
    print(f"Found {len(audio_files)} audio files.")
    all_paired_data = []  # Initialize an empty list to collect pairs from all files

    for audio_file in audio_files:
        if audio_file.lower().endswith('.wav'):
            print(f"Loading audio file: {audio_file}")
            label_file = os.path.splitext(audio_file)[0] + '.txt'
            print(f"Looking for label file: {label_file}")

            audio_file_path = os.path.join(audio_path, audio_file)
            label_file_path = os.path.join(labels_path, label_file)

            if os.path.exists(label_file_path):
                print(f"Found corresponding label file: {label_file}")
                audio, chord_data = load_audio_with_labels(audio_file_path, label_file_path, sample_rate)
                print(f"Loaded {len(chord_data)} labels")

                audio_chunks, chord_labels = segment_and_label(audio, chord_data, sample_rate, chunk_duration, overlap)
                print(f"Segmented audio into {len(audio_chunks)} chunks with corresponding labels")

                melspectrograms = extract_melspectrogram(audio_chunks, sample_rate, n_fft, hop_length, n_mels)
                print(f"Extracted Melspectrograms for all chunks")

                # Pair each Melspectrogram with its label and add to all_paired_data
                paired_data = list(zip(melspectrograms, chord_labels))
                all_paired_data.extend(paired_data)  # Add the pairs for this file to the main list
                print(f"Total pairs so far: {len(all_paired_data)}")

            else:
                print(f"Corresponding label file not found for audio: {audio_file}")
                continue  # Skip this file and continue with the next

    print(f"Finished processing. Total number of pairs: {len(all_paired_data)}")
    return all_paired_data  # Return all the paired data


def split_data(paired_data):
    print("Starting to split data into features and labels...")

    # Unzip the paired data into separate lists for melspectrograms and chord labels
    melspectrograms, chord_labels = zip(*paired_data)
    melspectrograms = list(melspectrograms)
    chord_labels = list(chord_labels)

    print("Unzipped data into features and labels.")

    # Calculate the number of unique chords
    unique_chords = set(chord_labels)
    num_unique_chords = len(unique_chords)
    print(f"Found {num_unique_chords} unique chords.")

    # Encode the chord labels into integers
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(chord_labels)

    # Print the mapping of original labels to encoded labels
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    print("Label mapping:", label_mapping)

    # Split the data into training, validation, and test sets
    print("Splitting the dataset into train, validation, and test sets...")
    x_train, x_temp, y_train, y_temp = train_test_split(melspectrograms, encoded_labels, test_size=0.2,
                                                        stratify=encoded_labels)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp)

    # Convert lists to numpy arrays
    print("Converting lists to numpy arrays for model training...")
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    print("Data splitting complete.")
    return x_train, y_train, x_val, y_val, x_test, y_test, num_unique_chords, label_encoder

def preprocess():
    paired_data = process_file()
    x_train, y_train, x_val, y_val, x_test, y_test, num_unique_chords, label_encoder = split_data(paired_data)
    return x_train, y_train, x_val, y_val, x_test, y_test, num_unique_chords, label_encoder

