import pretty_midi
import matplotlib.pyplot as plt
import numpy as np

def extract_pitch_sequence(midi_file_path):
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file_path)

    # Initialize lists to store pitches and their corresponding start times
    pitches = []
    start_times = []

    # Iterate through all instruments and their notes
    for instrument in midi_data.instruments:
        for note in instrument.notes:
            pitches.append(note.pitch)  # Absolute pitch
            start_times.append(note.start)

    # Sort by start times to maintain the sequence
    sorted_notes = sorted(zip(start_times, pitches))
    start_times, pitches = zip(*sorted_notes)

    return start_times, pitches


def visualize_pitch_sequence(start_times, pitches):
    # Convert MIDI note numbers to note names
    note_names = [pretty_midi.note_number_to_name(pitch) for pitch in pitches]

    plt.figure(figsize=(12, 6))
    plt.plot(start_times, pitches, marker='o', linestyle='-', color='b')

    # Annotate each point with the corresponding note name
    for i, (x, y) in enumerate(zip(start_times, pitches)):
        plt.text(x, y + 0.5, note_names[i], fontsize=9, ha='center')

    plt.title('Absolute Pitch Sequence from MIDI')
    plt.xlabel('Time (seconds)')
    plt.ylabel('MIDI Note Number (Absolute Pitch)')
    plt.grid(True)
    plt.show()


# Example usage
midi_file_path = "C:/School/Year 4 Sem 2/Thesis 2/Dataset/MTC-ANN-2.0.1/mid/NLB015569_01.mid"  # Replace with the actual path
start_times, pitches = extract_pitch_sequence(midi_file_path)
visualize_pitch_sequence(start_times, pitches)
