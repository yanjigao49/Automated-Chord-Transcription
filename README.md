# 🎵 CNN-Based Chord Transcription from Audio

This project presents a deep learning system for **automated chord recognition** from audio recordings, targeting a wide range of chord types—including standard, hybrid (slash), and poly chords. Built using a convolutional neural network (CNN) architecture, it transforms raw audio into meaningful chord annotations using mel-spectrogram analysis and a robust music theory-based representation framework.


## 🧠 Music Knowledge Representation Framework

### Note Class

- Attributes:
  - `letter` (A–G)
  - `modifier` (number of sharps/flats)
  - `pitch_class` (0–11)
  - `letter_class` (0–6)
- Methods:
  - `sharpen()`, `flatten()`
  - `enharmonic_up()`, `enharmonic_down()`, `enharmonic_respell()`
  - `transpose(interval, direction)`

### Chord Class

#### Regular Chord
- Attributes:
  - Root (Note)
  - Type (Triad, Seventh)
  - Inversion
  - Tensions
  - Chord Notes

#### Hybrid Chord
- Chord + independent bass note

#### Polychord
- UpperChord / LowerChord

All chord classes support:
- Transposition
- Normalization to unique flat-based forms

## 🧪 Model Architecture

- 4 convolutional layers + 2 fully connected layers
- Each Conv block:
  - Batch Normalization → ReLU → Pooling
- Pooling: MaxPool (first 3 layers), AvgPool (last)

### Layer Breakdown

1. CNN16 → MaxPool
2. CNN32 → MaxPool
3. CNN64 → MaxPool
4. CNN128 → AvgPool
5. Linear (128)
6. Linear (64)
7. Output (Chord label)

- All activations: ReLU
