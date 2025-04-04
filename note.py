import re

'''
Note Class
Attributes: Letter (Char), Modifier (Positive/Negative for # sharps/flats), Pitch Class(0-11), Letter Class(0-6)
Implements: EQComparable, Hashable
Methods: 
- sharpen (# of sharps to add, default 1)
- flatten (# of flats to add, default 1)
- enharmonic_up (Move # letters up without changing pitch, default 1)
- enharmonic_down (Move # letters down without changing pitch, default 1)
- enharmonic_respell (Reduce # of accidentals if possible, otherwise stay same or respell to their counterpart Ex C# -> Db)
- transpose (Interval, Direction) Distinguish between perfect and imperfect interval Ex: p4, P5, AaA5, m2, M2, d2
'''


class Note:
    # Define the default pitch classes for natural notes
    natural_notes_pitch_classes = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    # Define the sequence of notes for transposing up or down
    note_sequence = ['C', 'D', 'E', 'F', 'G', 'A', 'B']

    def __init__(self, letter, modifier):
        self.letter = letter.upper()  # Store the letter in uppercase
        self.modifier = modifier  # Store the modifier (sharp or flat)
        self.pitch_class = self.calculate_pitch_class()
        self.position = self.note_sequence.index(self.letter)  # Get the position of the letter in the sequence

    def calculate_pitch_class(self):
        # Calculate the pitch class based on the letter and modifier
        base_pitch_class = self.natural_notes_pitch_classes[self.letter]
        return (base_pitch_class + self.modifier) % 12

    def __str__(self):
        # Represent the note as a string with the letter and accidental
        accidental_str = ''
        if self.modifier > 0:
            accidental_str = '#' * self.modifier
        elif self.modifier < 0:
            accidental_str = 'b' * (-self.modifier)
        return f"{self.letter}{accidental_str}"

    def __eq__(self, other):
        """Compare all attributes of the Note object for equality."""
        if not isinstance(other, Note):
            # Don't attempt to compare against unrelated types
            return NotImplemented

        return self.pitch_class == other.pitch_class

    def __hash__(self):
        # A possible implementation of a hash function for the Note class
        return hash((self.letter, self.modifier, self.pitch_class, self.position))

    def sharpen(self, num_sharps=1):
        self.modifier += num_sharps
        self.pitch_class = (self.pitch_class + num_sharps) % 12

    def flatten(self, num_flats=1):
        self.modifier -= num_flats
        self.pitch_class = (self.pitch_class - num_flats) % 12

    def enharmonic_up(self, count=1):
        for _ in range(count):
            # Move one note up in the sequence
            self.position = (self.position + 1) % 7
            self.letter = self.note_sequence[self.position]
            # Adjust the modifier for enharmonic change
            self.modifier -= 1 if self.letter in ['C', 'F'] else 2

    def enharmonic_down(self, count=1):
        for _ in range(count):
            # Move one note down in the sequence
            self.position = (self.position - 1) % 7
            self.letter = self.note_sequence[self.position]
            # Adjust the modifier for enharmonic change
            self.modifier += 1 if self.letter in ['B', 'E'] else 2

    def enharmonic_respell(self):
        if self.modifier == 0:
            # It's a natural note, do nothing.
            return

        # Correct Fb, E#, Cb, B# to their natural equivalents
        if self.letter == 'F' and self.modifier == -1:
            self.letter = 'E'
            self.modifier = 0
        elif self.letter == 'E' and self.modifier == 1:
            self.letter = 'F'
            self.modifier = 0
        elif self.letter == 'C' and self.modifier == -1:
            self.letter = 'B'
            self.modifier = 0
        elif self.letter == 'B' and self.modifier == 1:
            self.letter = 'C'
            self.modifier = 0

        # For other letters with one sharp or flat
        elif self.modifier == 1:
            self.enharmonic_up()
        elif self.modifier == -1:
            self.enharmonic_down()

        # For notes with multiple accidentals
        while self.modifier > 1:
            self.enharmonic_up()
        while self.modifier < -1:
            self.enharmonic_down()

    def transpose(self, interval, direction):
        if direction not in [1, -1]:
            raise ValueError("Direction must be 1 (up) or -1 (down)")

        # Determine the interval number and quality from the string
        match = re.match(r'([mMPpAaDd]+)(\d+)', interval)
        if not match:
            raise ValueError("Invalid interval format")

        quality, number = match.groups()
        number = ((int(number) - 1) % 7) + 1

        # Check if the interval is perfect or imperfect
        perfect_intervals = [4, 5]
        is_perfect = number in perfect_intervals

        # Set the base offset depending on the interval number
        interval_distance = [0, 2, 4, 5, 7, 9, 11]
        steps = number - 1  # Number of steps to move in the note sequence
        offset = direction * interval_distance[steps % 7]

        # Adjust offset based on the interval quality
        if is_perfect:
            if 'a' in quality.lower():
                offset += direction * quality.lower().count('a')  # Augmented by the number of 'A's or 'a's
            elif 'd' in quality.lower():
                offset -= direction * quality.lower().count('d')  # Diminished by the number of 'd's or 'D's
        else:
            if quality == 'm':
                offset -= direction * 1  # Minor is one half-step lower than major
            elif 'a' in quality.lower():
                offset += direction * quality.lower().count('a')  # Augmented by the number of 'A's or 'a's
            elif 'd' in quality.lower():
                offset -= direction * (quality.lower().count(
                    'd') + 1)  # Diminished by the number of 'd's or 'D's and an extra half step

        # Transpose the note
        if direction > 0:
            self.enharmonic_up(steps)
        else:
            self.enharmonic_down(steps)

        # Apply Offset
        if offset > 0:
            self.sharpen(offset)
        else:
            self.flatten(-offset)


def note_from_string(note_str):
    # Determine the letter and accidental from the string
    letter = note_str[0].upper()
    accidental_str = note_str[1:]

    # Count the number of accidentals
    sharps = accidental_str.count('#')
    flats = accidental_str.count('b')

    # Calculate the modifier based on the number of sharps and flats
    modifier = sharps - flats

    # Create a new Note object
    return Note(letter, modifier)

