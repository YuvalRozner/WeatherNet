import numpy as np
import tensorflow as tf

def main():
    # Possibly enable unsafe deserialization if you trust your own code
    # tf.keras.config.enable_unsafe_deserialization()

    loaded_model = tf.keras.models.load_model("my_spatiotemporal_model.keras")
    print("Model loaded.")

    # Create dummy test
    L = 3
    S = 5
    num_features = 6
    X_test = np.random.rand(1, L, S, num_features).astype(np.float32)

    preds = loaded_model.predict(X_test)
    print("Predictions:", preds)

if __name__ == "__main__":
    main()
