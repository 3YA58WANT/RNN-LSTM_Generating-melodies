import json
import numpy as np
import tensorflow.keras as keras
import music21 as m21
from preprocess import SEQUENCE_LENGTH, MAPPING_PATH

class MelodyGenerator:

    def __init__(self, model_path="model.h5"):

        self.model = keras.models.load_model(model_path)

        with open(MAPPING_PATH, "r") as fp:
            self._mappings = json.load(fp)

        self._start_symbols = ["/"] * SEQUENCE_LENGTH

    def generate_melody(self, seed, num_steps, max_sequence_length, temperature):

        seed = seed.split()
        melody = seed
        seed = self._start_symbols + seed

        seed = [self._mappings[symbol] for symbol in seed]

        for _ in range(num_steps):

            seed = seed[-max_sequence_length:]

            onehot_seed = keras.utils.to_categorical(seed, num_classes=len(self._mappings))

            probabilities = self.model.predict(onehot_seed)[0]

            output_int = self._sample_with_temperature(probabilities, temperature)

            seed.append(output_int)
            melody.append([k for k, v in self._mappings.items() if v == output_int][0])

            if melody[-1] == "/":
                break

        return melody

    def _sample_with_temperature(self, probabilities, temperature):

        predictions = np.log(probabilities) / temperature
        probabilities = np.exp(predictions) / np.sum(np.exp(predictions))

        choices = range(len(probabilities))
        index = np.random.choice(choices, p=probabilities)

        return index

    def save_melody(self, melody, step_duration=0.25, format="midi", file_name="mel.mid"):

        stream = m21.stream.Stream()

        for symbol in melody:

            if symbol != "_":
                quarter_length_duration = step_duration

                if symbol == "r":
                    m21_event = m21.note.Rest(quarterLength=quarter_length_duration)
                else:
                    m21_event = m21.note.Note(int(symbol), quarterLength=quarter_length_duration)

                stream.append(m21_event)

        stream.write(format, file_name)

