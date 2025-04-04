from data_augmenter import *
from data_preprocesser import *
from formatter import format_time_data
from training import *
import pickle

# formatted_filename = format_time_data('input_data.txt')
# print(f"Formatted data has been written to {formatted_filename}")

#augment('AMW.wav','AMW.lab')
#augment('ANNA.wav','ANNA.lab')
#augment('ATOH.wav','ATOH.lab')
#augment('BIY.wav','BIY.lab')
#augment('BOYS.wav','BOYS.lab')
#augment('CHAINS.wav','CHAINS.lab')
#augment('DYWTKAS.wav','DYWTKAS.lab')
#augment('ISHST.wav','ISHST.lab')
#augment('LMD.wav','LMD.lab')
#augment('MSRY.wav','MSRY.lab')
#augment('PPM.wav','PPM.lab')
#augment('PSILY.wav','PSILY.lab')

train(32,100)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def load_model(model_path, num_unique_chords):
    model = AudioCNN(num_unique_chords)
    model.load_state_dict(torch.load('best_model_checkpoint.pth'))
    model.eval()
    model.to(device)
    return model

def load_label_encoder(file_name='label_encoder.pkl'):
    with open(file_name, 'rb') as file:
        label_encoder = pickle.load(file)
    return label_encoder

def load_num_unique_chords(file_name='num_unique_chords.pkl'):
    with open(file_name, 'rb') as file:
        num_unique_chords = pickle.load(file)
    return num_unique_chords

num_unique_chords = load_num_unique_chords()
model = load_model('best_model_checkpoint.pth',num_unique_chords)
label_encoder = load_label_encoder()

def combine_chords(chords):
    """
    Combine consecutive lines with the same chord label.
    :param chords: List of tuples containing start time, end time, and chord label.
    :return: Combined list of chords.
    """
    combined_chords = []
    current_chord = None

    for start, end, chord in chords:
        # If it's the first chord or a different chord, start a new segment
        if current_chord is None or chord != current_chord[2]:
            if current_chord is not None:
                combined_chords.append(current_chord)
            current_chord = [start, end, chord]
        else:
            # Extend the end time of the current chord
            current_chord[1] = end

    # Add the last chord
    if current_chord is not None:
        combined_chords.append(current_chord)

    return combined_chords

def combine_chords_from_file(input_file_path):
    """
    Read chords from a text file, combine consecutive lines with the same chord label,
    and write the combined chords to another text file in the same directory with '_condensed' added to the file name.
    :param input_file_path: Path to the input text file containing chords.
    """
    output_file_path = os.path.splitext(input_file_path)[0] + '_condensed.txt'

    with open(input_file_path, 'r') as input_file:
        chords = [tuple(line.strip().split()) for line in input_file.readlines()]
        chords = [(float(start), float(end), chord) for start, end, chord in chords]

    combined_chords = combine_chords(chords)

    with open(output_file_path, 'w') as output_file:
        for start, end, chord in combined_chords:
            output_file.write(f"{start:.2f} {end:.2f} {chord}\n")



def process_single_file(audio_file_name, model, label_encoder):
    """
    Process a single audio file, predict chords for each segment, and output audio-label pairs to a text file.
    :param audio_file_name: The name of the audio file to process.
    :param model: The trained neural network model for chord prediction.
    :param label_encoder: The LabelEncoder used for decoding model outputs to chord labels.
    :return: None
    """
    audio_file_path = os.path.join(test_path, audio_file_name)

    # Check if the audio file exists
    if not os.path.exists(audio_file_path):
        print(f"Audio file {audio_file_name} does not exist.")
        return

    # Load the audio file
    print(f"Loading audio file: {audio_file_name}")
    audio, _ = librosa.load(audio_file_path, sr=sample_rate)

    # Segment the audio
    print("Segmenting audio...")
    step_size = int((1 - overlap) * chunk_duration * sample_rate)
    segments = librosa.util.frame(audio, frame_length=int(chunk_duration * sample_rate),
                                  hop_length=step_size)

    # Extract Melspectrogram features
    print("Extracting Melspectrograms...")
    melspectrograms = [librosa.feature.melspectrogram(y=seg, sr=sample_rate, n_fft=n_fft,
                                                      hop_length=hop_length, n_mels=n_mels)
                       for seg in segments.T]

    # Convert to dataset and dataloader for the model
    dataset = AudioDataset(np.array(melspectrograms), np.zeros(len(melspectrograms)))  # Dummy labels
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Predict chords
    predicted_labels = []
    model.eval()
    with torch.no_grad():
        for features, _ in loader:
            features = features.to(device)
            outputs = model(features)
            _, predicted_indices = torch.max(outputs, 1)
            label = label_encoder.inverse_transform(predicted_indices.cpu().numpy())[0]
            predicted_labels.append(label)

    # Output the results to a text file
    output_file_name = os.path.splitext(audio_file_name)[0] + '_transcription.txt'
    output_file_path = os.path.join(test_path, output_file_name)
    with open(output_file_path, 'w') as txtfile:
        for i, label in enumerate(predicted_labels):
            start_time = i * step_size / sample_rate
            end_time = start_time + chunk_duration
            txtfile.write(f"{start_time:.2f} {end_time:.2f} {label}\n")

    combine_chords_from_file(output_file_path)

    print(f"Transcription written to {output_file_path}")

# process_single_file("DSB.wav",model,label_encoder)