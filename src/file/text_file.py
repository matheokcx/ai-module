from pathlib import Path
import tensorflow as tf
from src.const import BATCH_SIZE

DATA_DIR = Path(__file__).resolve().parents[2] / 'data'

class TextFile:
    def __init__(self, file_path):
        self.file_path = DATA_DIR / file_path

    def get_lines(self):
        with open(self.file_path) as file:
            lines = []
            for line in file.readlines():
                if line != "" and line != "\n":
                    lines.append(line.lower())
        return lines

    def transform_into_dataset(self):
        return (tf.data.Dataset.from_tensor_slices(self.get_lines())
                .batch(BATCH_SIZE)
                .shuffle(1000))
