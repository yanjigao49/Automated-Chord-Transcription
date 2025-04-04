from note import *
from enum import Enum

'''
Chord Class
Attributes: Root (Note), Type (Triad/Seventh Chord), Inversion(0-2/0-3), Chord Notes (Set of Note)
Implements: EQComparable
Methods: 
- transpose (Interval, Direction) Distinguish between perfect and imperfect interval Ex: p4, P5, AaA5, m2, M2, d2
'''


class ChordType(Enum):
    NO_CHORD = 0
    TRIAD = 1
    SEVENTH_CHORD = 2

# Define the chord intervals
triad_intervals = {
    '': ['P1', 'M3', 'P5'],  # Major
    'm': ['P1', 'm3', 'P5'],  # Minor
    'aug': ['P1', 'M3', 'A5'],  # Augmented
    'dim': ['P1', 'm3', 'D5'],  # Diminished
    'sus2': ['P1', 'M2', 'P5'],  # Sus2
    'sus4': ['P1', 'P4', 'P5'],  # Sus4
    'add9': ['P1', 'M3', 'P5', 'M9'],  # Add9 (Major)
    'madd9': ['P1', 'm3', 'P5', 'M9'],  # Add9 (Minor)
    '6': ['P1', 'M3', 'P5', 'M6'],  # 6 (Major)
    'm6': ['P1', 'm3', 'P5', 'M6'],  # 6 (Minor)
    '69': ['P1', 'M3', 'P5', 'M6', 'M9'],  # 69 (Major)
    'm69': ['P1', 'm3', 'P5', 'M6', 'M9']  # 69 (Minor)
}

seventh_chord_intervals = {
    'M7': ['P1', 'M3', 'P5', 'M7'],  # Major 7th
    'm7': ['P1', 'm3', 'P5', 'm7'],  # Minor 7th
    '7': ['P1', 'M3', 'P5', 'm7'],  # Dominant 7th
    'm7b5': ['P1', 'm3', 'D5', 'm7'],  # Half Diminished 7th
    'dim7': ['P1', 'm3', 'D5', 'D7'],  # Diminished 7th
    'augM7': ['P1', 'M3', 'A5', 'M7'],  # Augmented Major 7th
    'mM7': ['P1', 'm3', 'P5', 'M7'],  # Minor Major 7th
    'aug7': ['P1', 'M3', 'A5', 'm7'],  # Augmented 7th
    '7sus4': ['P1', 'P4', 'P5', 'm7'],  # 7sus4
}

# Tensions, treated as a set of numbers within parentheses for seventh chords
tension_intervals = {
    '9': 'M9',
    '11': 'P11',
    '13': 'M13',
    'b9': 'm9',
    '#9': 'A9',
    '#11': 'A11',
    'b13': 'm13'
}

valid_notes = ['Ab', 'A', 'Bb', 'B', 'C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G']

def _split_root_quality(chord_str):
    # Use regex to split the chord into root note and quality
    match = re.match(r"([A-Ga-g][#b]*)(.*)", chord_str)
    if not match:
        raise ValueError(f"Invalid chord format: '{chord_str}'")
    return match.groups()

class Chord:

    def __init__(self, chord_str, inversion=0, tensions=None):

        if chord_str.lower() == 'nc':
            self.type = ChordType.NO_CHORD
            return

        root_note_str, quality_str = _split_root_quality(chord_str)
        self.root_note = note_from_string(root_note_str)

        if quality_str in triad_intervals.keys():
            self.type = ChordType.TRIAD
        elif quality_str in seventh_chord_intervals.keys():
            self.type = ChordType.SEVENTH_CHORD
        else:
            raise ValueError(f"Chord quality '{quality_str}' is not recognized.")

        self.quality = quality_str

        if (self.type == ChordType.TRIAD and inversion not in range(3)) or \
                (self.type == ChordType.SEVENTH_CHORD and inversion not in range(4)):
            raise ValueError(f"Inversion '{inversion}' is not recognized for chord type '{self.type.name}'.")
        self.inversion = inversion

        # Check for tensions when the chord is a triad
        if tensions and self.type != ChordType.SEVENTH_CHORD:
            raise ValueError(f"Tensions are not supported for non 7th chords. Provided tensions: {tensions}")

        self.tensions = tensions or []

        self.notes = self.calculate_chord_notes()  # Store notes as a set

    def calculate_chord_notes(self):
        # Determine which interval set to use based on the chord quality
        if self.quality in triad_intervals:
            interval_set = triad_intervals[self.quality]
        elif self.quality in seventh_chord_intervals:
            interval_set = seventh_chord_intervals[self.quality]
        else:
            raise ValueError(f"Chord quality '{self.quality}' is not recognized.")

        # Create a new set for the notes
        chord_tones = set()

        # Take the root note, transpose it to each interval, and add to the set
        for interval in interval_set:
            transposed_note = note_from_string(str(self.root_note))  # Create a new copy of the root note
            transposed_note.transpose(interval, 1)  # Transpose by the interval
            chord_tones.add(transposed_note)

        # Add tensions to the chord tones if they are provided

        for tension in self.tensions:
            if tension in tension_intervals:
                tension_interval = tension_intervals[tension]
                tension_note = note_from_string(str(self.root_note))  # Create a new copy of the root note
                tension_note.transpose(tension_interval, 1)  # Transpose by the tension interval
                chord_tones.add(tension_note)
            else:
                raise ValueError(f"Tension '{tension}' is not recognized.")

        return chord_tones

    def normalize(self):
        if self.type == ChordType.NO_CHORD:
            return

        while self.root_note.__str__() not in valid_notes:
            print(f"Normalizing Chord: {self.root_note}")  # Print statement added here
            # respell the chord enharmonically
            self.root_note.enharmonic_respell()
            for note in self.notes:
                note.enharmonic_respell()

    def transpose(self, interval, direction):
        if self.type == ChordType.NO_CHORD:
            return

        # Transpose the root note
        self.root_note.transpose(interval, direction)

        # Recalculate the notes
        for note in self.notes:
            note.transpose(interval, direction)

    def __str__(self):
        if self.type == ChordType.NO_CHORD:
            return "NC"
        # Start with the root note and quality
        chord_str = f"{self.root_note}{self.quality}"

        # Include the tensions in parentheses, if there are any
        if self.tensions:
            tensions_str = ','.join(str(tension) for tension in self.tensions)
            chord_str += f"({tensions_str})"

        # Append the inversion information
        if self.inversion == 1:
            chord_str += "/3"
        elif self.inversion == 2:
            chord_str += "/5"
        elif self.inversion == 3:
            chord_str += "/7"

        return chord_str

    def __eq__(self, other):
        """Chords are equal if the symmetric difference of their notes is empty."""
        if not isinstance(other, Chord):
            # Don't attempt to compare against unrelated types
            return NotImplemented

        # Calculate the symmetric difference of the two sets of notes
        notes_in_self_xor_other = self.notes ^ other.notes
        # Two chords are equal if the symmetric difference is an empty set
        return not notes_in_self_xor_other


'''
Hybrid Chord Class
Attributes: Chord (Chord), Bass Note(Note), Chord Notes (Set of Note)
Implements: EQComparable
Methods:
- transpose (Interval, Direction) Distinguish between perfect and imperfect interval Ex: p4, P5, AaA5, m2, M2, d2
'''


class HybridChord(Chord):
    def __init__(self, chord_str, bass_note_str, tensions=None):
        # Initialize with the base Chord class, always with inversion 0
        super().__init__(chord_str, inversion=0, tensions=tensions)
        # Add the extra bass note
        self.bass_note = note_from_string(bass_note_str)
        # Add the bass note to the chord's notes set
        self.notes.add(Note(self.bass_note.letter,self.bass_note.modifier))

    def transpose(self, interval, direction):
        # Transpose the root note
        self.root_note.transpose(interval, direction)
        self.bass_note.transpose(interval, direction)

        # Recalculate the notes
        for note in self.notes:
            note.transpose(interval, direction)

    def normalize(self):
        print(f"Normalizing HybridChord: {self}")  # Print statement added here
        while self.root_note.__str__() not in valid_notes:
            self.root_note.enharmonic_respell()
            for note in self.notes:
                note.enharmonic_respell()
        while self.bass_note.__str__() not in valid_notes:
            self.bass_note.enharmonic_respell()

    def __str__(self):
        # Include the bass note after the chord representation
        chord_str = super().__str__()
        return f"{chord_str}/{self.bass_note}"

    def __eq__(self, other):
        # Check if other is an instance of HybridChord
        if not isinstance(other, HybridChord):
            return NotImplemented

        # Two HybridChords are equal if they have the same bass note and the same set of notes
        return self.bass_note == other.bass_note and self.notes == other.notes


'''
Polychord Class
Attributes: UpperChord, LowerChord, Chord Notes (Set of Note)
Implements: EQComparable
Methods:
- transpose (Interval, Direction) Distinguish between perfect and imperfect interval Ex: p4, P5, AaA5, m2, M2, d2
'''

class Polychord:
    def __init__(self, upper_chord_str, lower_chord_str):
        # Initialize the upper and lower chords, both in root position (inversion=0)
        self.upper_chord = Chord(upper_chord_str, inversion=0)
        self.lower_chord = Chord(lower_chord_str, inversion=0)
        self.notes = self.upper_chord.notes | self.lower_chord.notes

    def transpose(self, interval, direction):
        self.upper_chord.transpose(interval,direction)
        self.lower_chord.transpose(interval,direction)
        self.notes = self.upper_chord.notes | self.lower_chord.notes

    def normalize(self):
        print(f"Normalizing Polychord: {self}")  # Print statement added here
        while self.upper_chord.root_note not in valid_notes:
            self.upper_chord.normalize()
        while self.lower_chord.root_note not in valid_notes:
            self.lower_chord.normalize()

    def __eq__(self, other):
        # Check if other is an instance of Polychord
        if not isinstance(other, Polychord):
            return NotImplemented

        # Two Polychords are equal if the root of the lower chords are the same
        # and the union of the upper and lower chord notes are equal
        return (self.lower_chord.root_note == other.lower_chord.root_note and
                (self.upper_chord.notes | self.lower_chord.notes) ==
                (other.upper_chord.notes | other.lower_chord.notes))

    def __str__(self):
        # Represent the polychord as the upper chord over the lower chord
        return f"{self.upper_chord}|{self.lower_chord}"


def parse_chord_string(chord_string):
    if chord_string.lower() == "nc":
        return Chord("NC")

    # Match Polychord first (e.g., "Cm7|G#")
    polychord_match = re.match(r'(.+)\|(.+)', chord_string)
    if polychord_match:
        upper_chord_str, lower_chord_str = polychord_match.groups()
        return Polychord(upper_chord_str.strip(), lower_chord_str.strip())

    # Match Hybrid Chord (e.g., "Cm7/G")
    hybrid_match = re.match(r'(.+)/([A-Ga-g][#b]*)$', chord_string)
    if hybrid_match:
        chord_str, bass_note_str = hybrid_match.groups()
        return HybridChord(chord_str.strip(), bass_note_str.strip())

    # Match regular Chord with optional tensions and inversion (e.g., "Cm7(b9,13)/5")
    chord_match = re.match(r'([A-Ga-g][#b]*)', chord_string)
    if chord_match:
        # Split tensions and inversion from chord
        root_quality, *extensions_inversion = re.split(r'(\(.+\)|/\d)', chord_string[len(chord_match.group()):], 1)

        # Default values
        tensions = None
        inversion = 0

        # Process extensions and inversion if they exist
        if extensions_inversion:
            for part in extensions_inversion:
                if part.startswith('(') and part.endswith(')'):
                    tensions = part[1:-1].split(',')
                elif part.startswith('/'):
                    inversion = {'/3': 1, '/5': 2, '/7': 3}.get(part, 0)

        # Create and return the Chord object
        return Chord(chord_match.group() + root_quality.strip(), inversion=inversion, tensions=tensions)

    raise ValueError("The given chord string does not match any known chord patterns.")

# Checks if all the notes of chord1 are a subset of the notes in chord2.
def isCompatible(chord1, chord2):

    # Ensure both arguments are of a compatible type (Chord, HybridChord, Polychord)
    if not isinstance(chord1, (Chord, HybridChord, Polychord)) or \
            not isinstance(chord2, (Chord, HybridChord, Polychord)):
        raise TypeError("Arguments must be of type Chord, HybridChord, or Polychord.")

    # Check if chord1's notes are a subset of chord2's notes
    return chord1.notes.issubset(chord2.notes)
