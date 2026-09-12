import tensorflow as tf
from src.const import VOCAB_SIZE, MAX_LEN, EMBEDDING_DIM, N_HEADS, KEY_DIM, FEED_FORWARD_DIM, DROPOUT_RATE
from src.file.text_file import TextFile
from tensorflow.keras import layers, models, losses
from src.transformer.embedding_layer import EmbeddingLayer
from src.transformer.transformer import Transformer

file = TextFile('shakspeare.txt')
inputs = layers.Input(shape=(None,), dtype=tf.int32)
x = EmbeddingLayer(MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM)(inputs)
x, attention_scores = Transformer(N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM, DROPOUT_RATE)(x)
outputs = layers.Dense(VOCAB_SIZE, activation='softmax')(x)
gpt = models.Model(inputs = inputs, outputs = [outputs, attention_scores])
gpt.compile('adam', loss = [losses.SparseCategoricalCrossentropy(), None])
gpt.summary()

'''
app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

app.run(
    host = 'localhost',
    port = 5000,
    debug = True
)
'''
