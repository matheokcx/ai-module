VOCAB_SIZE = 5000

N_HEADS = 2

# Taille du context
MAX_LEN = 80

EMBEDDING_DIM = 256
KEY_DIM = 256

FEED_FORWARD_DIM = 256

BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2

SEED = 42
EPOCHS = 10

# Décalage entre deux fenêtres d'entraînement (< MAX_LEN => fenêtres qui se recouvrent)
STRIDE = MAX_LEN // 2

DROPOUT_RATE = 0.1
