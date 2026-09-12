import tensorflow as tf
from tensorflow.keras import layers

class Transformer(layers.Layer):
    def __init__(self,  num_heads, key_dim, embed_dim, ff_dim, dropout_rate):
        super(Transformer, self).__init__()
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        # Couches de la structure du transformer
        self.attn = layers.MultiHeadAttention(
            num_heads = num_heads,
            key_dim = key_dim,
            output_shape=embed_dim
        )
        self.dropout_1 = layers.Dropout(self.dropout_rate)
        self.ln_1 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn_1 = layers.Dense(self.ff_dim, activation='relu')
        self.ffn_2 = layers.Dense(self.embed_dim)
        self.dropout_2 = layers.Dropout(self.dropout_rate)
        self.ln_2 = layers.LayerNormalization(epsilon=1e-6)

    def causal_attention_mask(self, batch_size, dest_token_nb, source_token_nb, type):
        i = tf.range(dest_token_nb)[:, None]
        j = tf.range(source_token_nb)
        m = i >= j - source_token_nb + dest_token_nb  # Détermination de la relation entre les indices/token

        mask = tf.cast(m, type)
        mask = tf.reshape(mask, [1, dest_token_nb, source_token_nb])

        mult = tf.concat([
            tf.expand_dims(batch_size, -1),
            tf.constant([1, 1], dtype=tf.int32),
        ], axis = 0)
        return tf.tile(mask, mult)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seg_len = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, seg_len, seg_len, tf.bool)
        attention_output, attention_scores = self.attn(
            inputs,
            inputs,
            attention_mask = causal_mask,
            return_attention_scores = True
        )
        attention_output = self.dropout_1(attention_output)
        out1 = self.ln_1(inputs + attention_output)
        ffn_1 = self.ffn_1(out1)
        ffn_2 = self.ffn_2(ffn_1)
        ffn_output = self.dropout_2(ffn_2)
        return (self.ln_2(out1 + ffn_output), attention_scores)
