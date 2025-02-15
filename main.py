import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from mido import MidiFile
import os
import matplotlib.pyplot as plt

# Step 1: Load MIDI and Extract Pitch Data
def extract_midi_notes(midi_file):
    midi = MidiFile(midi_file)
    notes = []
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'note_on':
                notes.append(msg.note)
    return notes

# Convert MIDI numbers to note names
def midi_to_note_name(midi_number):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi_number // 12) - 1
    note = note_names[midi_number % 12]
    return f"{note}{octave}"

# Step 2: Convert to Relative Pitch Sequence
def convert_to_relative_sequence(notes):
    return np.diff(notes)

# Step 3: Generate N-grams
def generate_ngrams(sequence, n=4):
    return [tuple(sequence[i:i+n]) for i in range(len(sequence) - n + 1)]

# Step 4: Edit Distance Calculation
def edit_distance(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = np.zeros((m+1, n+1), dtype=int)

    for i in range(m+1):
        for j in range(n+1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]

# Step 5: Bipartite Graph Matching
def bipartite_graph_matching(ngrams_a, ngrams_b):
    cost_matrix = np.zeros((len(ngrams_a), len(ngrams_b)))

    for i, gram_a in enumerate(ngrams_a):
        for j, gram_b in enumerate(ngrams_b):
            cost_matrix[i, j] = edit_distance(gram_a, gram_b)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()

    return total_cost, cost_matrix

# Step 6: Normalize and Calculate Similarity Score
def normalize_cost(cost_matrix):
    return np.log1p(cost_matrix) / np.log1p(cost_matrix.max())

def similarity_score(total_cost, max_possible_cost):
    return 1 - (total_cost / max_possible_cost)

# Step 7: Evaluation Metrics
def evaluate_model(y_true, y_scores):
    auc = roc_auc_score(y_true, y_scores)
    map_score = average_precision_score(y_true, y_scores)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return auc, map_score, fpr, tpr

# Example Usage
midi_file1 = "C:/School/Year 4 Sem 2/Thesis 2/Dataset/MTC-ANN-2.0.1/mid/NLB015569_01.mid"  # Replace with your MIDI file path
midi_file2 = "C:/School/Year 4 Sem 2/Thesis 2/Dataset/MTC-ANN-2.0.1/mid/NLB070033_01.mid"  # Replace with your MIDI file path

query_song_notes = extract_midi_notes(midi_file1)
compare_song_notes = extract_midi_notes(midi_file2)

# Convert MIDI numbers to note names for visualization
query_note_names = [midi_to_note_name(note) for note in query_song_notes]
compare_note_names = [midi_to_note_name(note) for note in compare_song_notes]

# Visualizing Absolute Pitch Sequence with Note Names
plt.figure(figsize=(12, 6))
plt.plot(query_song_notes, label='Query Song', marker='o')
plt.plot(compare_song_notes, label='Compare Song', marker='x')
plt.title('Absolute Pitch Sequence with Note Names')
plt.xlabel('Note Index')
plt.ylabel('MIDI Number')
plt.xticks(ticks=range(len(query_note_names)), labels=query_note_names, rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

query_relative = convert_to_relative_sequence(query_song_notes)
compare_relative = convert_to_relative_sequence(compare_song_notes)

# Visualizing Relative Pitch Interval
plt.figure(figsize=(10, 4))
plt.plot(query_relative, label='Query Song')
plt.plot(compare_relative, label='Compare Song')
plt.title('Relative Pitch Interval')
plt.xlabel('Interval Index')
plt.ylabel('Pitch Interval')
plt.legend()
plt.show()

query_ngrams = generate_ngrams(query_relative)
compare_ngrams = generate_ngrams(compare_relative)

# Display N-grams
print(f"N-grams used (Query Song): {query_ngrams}")
print(f"N-grams used (Compare Song): {compare_ngrams}")

total_cost, cost_matrix = bipartite_graph_matching(query_ngrams, compare_ngrams)
normalized_cost_matrix = normalize_cost(cost_matrix)

# Visualizing Cost Matrix
plt.figure(figsize=(8, 6))
plt.imshow(cost_matrix, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Cost Matrix')
plt.xlabel('Compare Song N-grams')
plt.ylabel('Query Song N-grams')
plt.show()

# Visualizing Log Transformed Values
plt.figure(figsize=(8, 6))
plt.imshow(normalized_cost_matrix, cmap='plasma', aspect='auto')
plt.colorbar()
plt.title('Log Transformed Cost Matrix')
plt.xlabel('Compare Song N-grams')
plt.ylabel('Query Song N-grams')
plt.show()

similarity = similarity_score(total_cost, len(query_ngrams) * len(compare_ngrams))
print(f"Similarity Score: {similarity:.4f}")

# Simulate true labels and scores for ROC and MAP (For demonstration)
y_true = [1 if i < len(query_ngrams) / 2 else 0 for i in range(len(query_ngrams))]
y_scores = [1 - score for score in normalized_cost_matrix.mean(axis=1)]

auc, map_score, fpr, tpr = evaluate_model(y_true, y_scores)

# Visualizing ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.show()

print(f"AUC: {auc:.4f}")
print(f"MAP: {map_score:.4f}")
