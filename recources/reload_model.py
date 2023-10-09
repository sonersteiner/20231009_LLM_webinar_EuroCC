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

encoded -= 2  # drop tokens 0 (pad) and 1 (unknown), which we will not use
n_tokens = text_vec_layer.vocabulary_size() - 2  # number of distinct chars = 39
dataset_size = len(encoded) 


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

checkpoint_path = "shakes_model/cp.ckpt"
#checkpoint_path = "my_shakespeare_model2/cp.ckpt" 
checkpoint_dir = os.path.dirname(checkpoint_path)

# Loads the weights
shakes_model.load_weights(checkpoint_path)
