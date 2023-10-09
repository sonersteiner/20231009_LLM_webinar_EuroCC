import os
import tensorflow as tf

tf.random.set_seed(42) 

def create_model():
	model = tf.keras.Sequential([
    	tf.keras.layers.Embedding(input_dim=n_tokens, output_dim=16),
   	 tf.keras.layers.GRU(128, return_sequences=True),
    	tf.keras.layers.Dense(n_tokens, activation="softmax")
	])

	model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
                      metrics=["accuracy"])
	
	return model

shakes_model = create_model()
shakes_model.summary()

checkpoint_path = "shakes_model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
# that is very important
cp_callback = tf.keras.callbacks.ModelCheckpoint(
              filepath=checkpoint_path,
              monitor="val_accuracy",
              save_best_only=True,
              save_weights_only=True,
              verbose=1)

# Train the model with the new callback
shakes_model.fit(train_set, 
          validation_data=valid_set,  
          epochs=3,
          callbacks=[cp_callback])  # Pass callback to training

