from pathlib import Path
import tensorflow as tf
from src.const import VOCAB_SIZE, MAX_LEN, EMBEDDING_DIM, N_HEADS, KEY_DIM, FEED_FORWARD_DIM, DROPOUT_RATE, EPOCHS, BATCH_SIZE, STRIDE
from src.file.text_file import TextFile
from tensorflow.keras import layers, models, losses, callbacks
from src.transformer.embedding_layer import EmbeddingLayer
from src.transformer.gpt import GPT
from src.transformer.transformer import Transformer

file = TextFile('shakspeare.txt')
inputs = layers.Input(shape=(None,), dtype=tf.int32)
x = EmbeddingLayer(MAX_LEN, VOCAB_SIZE, EMBEDDING_DIM)(inputs)
x, attention_scores = Transformer(N_HEADS, KEY_DIM, EMBEDDING_DIM, FEED_FORWARD_DIM, DROPOUT_RATE)(x)
outputs = layers.Dense(VOCAB_SIZE, activation='softmax')(x)
gpt = models.Model(inputs = inputs, outputs = [outputs, attention_scores])
gpt.compile('adam', loss = [losses.SparseCategoricalCrossentropy(), None])
gpt.summary()

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_PATH = str(ROOT / 'data' / 'w.weights.h5')
LOGS_DIR = str(ROOT / 'logs')

if Path(WEIGHTS_PATH).exists():
    gpt.load_weights(WEIGHTS_PATH)

model_checkpoint_callback = callbacks.ModelCheckpoint(
    filepath = WEIGHTS_PATH,
    save_weights_only = True,
    save_freq = 'epoch',
    verbose = 0
)
tensorboard_callback = callbacks.TensorBoard(log_dir = LOGS_DIR)


vectorize_layer = layers.TextVectorization(
    standardize = 'lower',
    max_tokens = VOCAB_SIZE,
    output_mode = 'int'
)
vectorize_layer.adapt(file.transform_into_dataset())
text_generator = GPT(vectorize_layer.get_vocabulary())
token_stream = vectorize_layer(tf.constant([' '.join(file.get_lines())]))[0]
n_windows = (int(tf.size(token_stream)) - (MAX_LEN + 1)) // STRIDE + 1
train_ds = (
    tf.data.Dataset.from_tensor_slices(token_stream)
    .window(MAX_LEN + 1, shift = STRIDE, drop_remainder = True)
    .flat_map(lambda window: window.batch(MAX_LEN + 1))
    .apply(tf.data.experimental.assert_cardinality(n_windows))
    .map(lambda chunk: (chunk[:-1], chunk[1:]))
    .shuffle(1000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
gpt.fit(
    train_ds,
    epochs = EPOCHS,
    callbacks = [model_checkpoint_callback, tensorboard_callback, text_generator]
)

text_generator.generate('Who deserves', max_tokens = 1000, temperature = 0.5)

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
