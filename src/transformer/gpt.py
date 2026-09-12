import numpy as np
from tensorflow.keras import callbacks
from src.const import MAX_LEN

class GPT(callbacks.Callback):
    def __init__(self, index_to_word, top_k = 10):
        super(GPT, self).__init__()
        self.index_to_word = index_to_word
        self.word_to_index = {
            word: index for index, word in enumerate(index_to_word)
        }

    def sample_from(self, probs, temperature=1.0):
        """
        Devine le prochain mot à partir des probabilités générées
        """
        probs = probs ** (1 / temperature)
        probs = probs / np.sum(probs)
        return np.random.choice(len(probs), p = probs), probs

    def generate(self, start_prompt, max_tokens, temperature=1.0):
        start_tokens = [self.word_to_index.get(x, 1) for x in start_prompt.split()]
        sample_token = None
        info = []

        while len(start_tokens) < max_tokens and sample_token != 0:
            # Le pos_emb ne connait que MAX_LEN positions : on ne donne au modele
            # que la fin du contexte, ce qui permet de generer au-dela de MAX_LEN.
            x = np.array([start_tokens[-MAX_LEN:]])
            y, att = self.model.predict(x, verbose = 0)
            sample_token, probs = self.sample_from(y[0][-1], temperature)
            info.append({
                "prompt": start_prompt,
                "word_probs": probs,
                "atts": att[0, :, -1, :]
            })
            start_tokens.append(sample_token)
            start_prompt = start_prompt + ' ' + self.index_to_word[sample_token]

        print(f"\nTexte généré: {start_prompt}")
        return info

    def on_epoch_end(self, epoch, logs = None):
        self.generate('my lord', max_tokens = 80, temperature = 1.0)

