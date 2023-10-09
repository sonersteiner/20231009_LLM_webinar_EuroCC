import os
import tensorflow as tf

tf.random.set_seed(42)
filepath="shakespeare.txt"
with open(filepath) as f:
    shakes_text = f.read()

text_vec_layer = tf.keras.layers.TextVectorization(split="character",
                                                   standardize="lower")
text_vec_layer.adapt([shakes_text])
encoded = text_vec_layer([shakes_text])[0]

encoded -= 2
n_tokens = text_vec_layer.vocabulary_size() - 2
dataset_size = len(encoded)

checkpoint_path = "shakes_model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Loads the weights
shakes_model.load_weights(checkpoint_path)

def create_model():
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dense(n_tokens, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",metrics=["accuracy"])
    return model

shakes_model = create_model()
shakes_model.summary()


# Loads the weights
shakes_model.load_weights(checkpoint_path)

log_probas = tf.math.log([[0.5, 0.4, 0.1]])  # probas = 50%, 40%, and 10%
tf.random.set_seed(42)
tf.random.categorical(log_probas, num_samples=8)  # draw 8 samples

def next_char(text, temperature=1):
    y_proba = shakes_model.predict([text])[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    return text_vec_layer.get_vocabulary()[char_id + 2]

def next_char_ss(text, my_model, temperature=1):
    y_proba = my_model.predict([text])[0, -1:]
    rescaled_logits = tf.math.log(y_proba) / temperature
    char_id = tf.random.categorical(rescaled_logits, num_samples=1)[0, 0]
    return text_vec_layer.get_vocabulary()[char_id + 2]

def extend_text(text, my_model, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text += next_char_ss(text, my_model, temperature)
    return text

print(extend_text("To be or not to be", shakes_model, temperature=0.5))
print(extend_text("To be or not to be",shakes_model))

