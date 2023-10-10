import os
import json
import music21 as m21
import numpy as np
import tensorflow.keras as keras


KERN_DATASET_PATH = "deutschl/erk"
SAVE_DIR = "dataset"
SINGLE_FILE_DATASET = "file_dataset"
MAPPING_PATH = "mapping.json"
SEQUENCE_LENGTH = 64


ACCEPTABLE_DURATIONS = [0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4]


def load_songs_in_kern(dataset_path):
    return [m21.converter.parse(os.path.join(path, file)) for path, _, files in os.walk(dataset_path) for file in files if file[-3:] == "krn"]


def has_acceptable_durations(song, acceptable_durations):
    return all(note.duration.quarterLength in acceptable_durations for note in song.flat.notesAndRests)


def transpose(song):
    key = song.analyze("key")
    interval = m21.interval.Interval(key.tonic, m21.pitch.Pitch("C")) if key.mode == "major" else m21.interval.Interval(key.tonic, m21.pitch.Pitch("A"))
    return song.transpose(interval)


def encode_song(song, time_step=0.25):
    encoded_song = []
    for event in song.flat.notesAndRests:
        if isinstance(event, m21.note.Note):
            symbol = event.pitch.midi
        elif isinstance(event, m21.note.Rest):
            symbol = "r"
        steps = int(event.duration.quarterLength / time_step)
        encoded_song.extend([symbol] + ["_"] * (steps - 1))
    return " ".join(map(str, encoded_song))


def preprocess(dataset_path):
    songs = load_songs_in_kern(dataset_path)
    songs = [transpose(song) for song in songs if has_acceptable_durations(song, ACCEPTABLE_DURATIONS)]
    for i, song in enumerate(songs):
        encoded_song = encode_song(song)
        save_path = os.path.join(SAVE_DIR, str(i))
        with open(save_path, "w") as fp:
            fp.write(encoded_song)
        if i % 10 == 0:
            print(f"Song {i} out of {len(songs)} processed")


def load(file_path):
    with open(file_path, "r") as fp:
        song = fp.read()
    return song


def create_single_file_dataset(dataset_path, file_dataset_path, sequence_length):
    new_song_delimiter = "/ " * sequence_length
    songs = " ".join(load(os.path.join(path, file)) for path, _, files in os.walk(dataset_path) for file in files)
    with open(file_dataset_path, "w") as fp:
        fp.write(songs + " " + new_song_delimiter)
    return songs


def create_mapping(songs, mapping_path):
    mappings = {symbol: i for i, symbol in enumerate(set(songs.split()))}
    with open(mapping_path, "w") as fp:
        json.dump(mappings, fp, indent=4)


def convert_songs_to_int(songs):
    with open(MAPPING_PATH, "r") as fp:
        mappings = json.load(fp)
    return [mappings[symbol] for symbol in songs.split()]


def generate_training_sequences(sequence_length):
    songs = load(SINGLE_FILE_DATASET)
    int_songs = convert_songs_to_int(songs)
    inputs = [int_songs[i:i+sequence_length] for i in range(len(int_songs) - sequence_length)]
    targets = np.array(int_songs[sequence_length:])
    vocabulary_size = len(set(int_songs))
    inputs = keras.utils.to_categorical(inputs, num_classes=vocabulary_size)
    print(f"There are {len(inputs)} sequences.")
    return inputs, targets


def main():
    preprocess(KERN_DATASET_PATH)
    songs = create_single_file_dataset(SAVE_DIR, SINGLE_FILE_DATASET, SEQUENCE_LENGTH)
    create_mapping(songs, MAPPING_PATH)
    #inputs, targets = generate_training_sequences(SEQUENCE_LENGTH)


if __name__ == "__main__":
    main()
