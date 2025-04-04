from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np


melspectrograms = [...]  # List of Melspectrogram arrays
labels = [...]  # Corresponding list of chord labels

# Encode the chord labels into a unique integer
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the dataset into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(melspectrograms, encoded_labels, test_size=0.2, stratify=encoded_labels)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

# Convert to numpy arrays for training
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)
X_test = np.array(X_test)
y_test = np.array(y_test)
