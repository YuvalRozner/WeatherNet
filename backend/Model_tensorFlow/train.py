#!/usr/bin/env python
# train_no_lambda.py

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input


###############################################################################
# 1) TRANSFORMER HELPER: ENCODER LAYER
###############################################################################
def transformer_encoder(inputs, d_model, num_heads, d_ff, dropout=0.1):
    """
    A single Transformer encoder layer.
    inputs: (batch, sequence_length, d_model)
    returns: (batch, sequence_length, d_model)
    """
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout
    )(inputs, inputs)
    out1 = layers.LayerNormalization(epsilon=1e-6)(inputs + attn_output)

    ffn = tf.keras.Sequential([
        layers.Dense(d_ff, activation='relu'),
        layers.Dense(d_model)
    ])
    ffn_output = ffn(out1)
    out2 = layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return out2


###############################################################################
# 2) STATION-LEVEL TRANSFORMER BLOCK (re-usable)
###############################################################################
class StationTransformer(layers.Layer):
    """
    Learns spatial dependencies among stations at a single time step.
    Input shape: (batch, S, d_model)
    Output shape: (batch, S, d_model)
    """
    def __init__(self, d_model=32, num_heads=2, d_ff=64, num_layers=1, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.enc_layers = []
        for _ in range(num_layers):
            self.enc_layers.append(
                lambda x: transformer_encoder(x, d_model, num_heads, d_ff, dropout)
            )

    def call(self, inputs, training=None):
        x = inputs  # shape: (batch, S, d_model)
        for enc in self.enc_layers:
            x = enc(x)
        return x  # (batch, S, d_model)


###############################################################################
# 3) TIME-LEVEL TRANSFORMER BLOCK (re-usable)
###############################################################################
class TimeTransformer(layers.Layer):
    """
    Learns temporal dependencies across L time steps for the target station.
    Input shape: (batch, L, d_model)
    Output shape: (batch, L, d_model)
    """
    def __init__(self, d_model=32, num_heads=2, d_ff=64, num_layers=1, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.dropout = dropout
        self.enc_layers = []
        for _ in range(num_layers):
            self.enc_layers.append(
                lambda x: transformer_encoder(x, d_model, num_heads, d_ff, dropout)
            )

    def call(self, inputs, training=None):
        x = inputs  # shape: (batch, L, d_model)
        for enc in self.enc_layers:
            x = enc(x)
        return x  # (batch, L, d_model)


###############################################################################
# 4) CUSTOM LAYER: STATION-SPATIAL ENCODER
#    - Loops over L time steps, loops over S stations
#    - Per-station CNN, then station-level Transformer
#    - Extract target station embedding
#    - Stacks them => (batch, L, d_model)
###############################################################################
class StationSpatialEncoder(layers.Layer):
    """
    Takes shape (batch, L, S, features).
    For each time step t in [0..L), does:
      - For each station s in [0..S), slice (batch, features).
      - Expand dims => (batch, 1, features), apply CNN => (batch, d_model).
      - Stack stations => (batch, S, d_model), feed to station-level transformer.
      - Extract station index 0 => (batch, d_model).
    Finally, stack across time => (batch, L, d_model).
    """
    def __init__(self,
                 L=3,
                 S=5,
                 d_model=32,
                 num_heads=2,
                 d_ff=64,
                 dropout=0.1,
                 num_station_transformer_layers=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.L = L
        self.S = S
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout
        self.num_station_transformer_layers = num_station_transformer_layers

        # Define the station-level CNN layers:
        # We'll keep it simple: one Conv1D + a GlobalAveragePooling
        # Because the time dimension for CNN is basically 1, it's mostly illustrative.
        self.cnn_conv = layers.Conv1D(filters=d_model, kernel_size=3,
                                      padding='causal', activation='relu')
        self.cnn_pool = layers.GlobalAveragePooling1D()

        # Define the station-level Transformer sub-block
        self.spatial_transformer = StationTransformer(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_station_transformer_layers,
            dropout=dropout
        )

    def call(self, inputs, training=None):
        """
        inputs: (batch, L, S, features)
        returns: (batch, L, d_model)
        """
        batch_size = tf.shape(inputs)[0]

        # We'll accumulate the target-station embedding at each time step here
        time_embeddings = []

        # Python loop over the time steps
        for t in range(self.L):
            # Extract shape => (batch, S, features)
            x_t = inputs[:, t, :, :]  # no Lambda needed, just direct slicing

            # Build a list of station embeddings
            station_embeds = []
            for s in range(self.S):
                # shape => (batch, features)
                x_station = x_t[:, s, :]

                # Expand dims to simulate (batch, time=1, features)
                x_station_expanded = tf.expand_dims(x_station, axis=1)  # shape: (batch, 1, features)

                # CNN
                x_cnn = self.cnn_conv(x_station_expanded)        # (batch, 1, d_model) if kernel_size=3
                x_cnn = self.cnn_pool(x_cnn)                     # (batch, d_model)

                # We'll keep shape (batch, 1, d_model) so we can stack stations
                x_cnn = tf.expand_dims(x_cnn, axis=1)            # (batch, 1, d_model)

                station_embeds.append(x_cnn)

            # Concatenate across stations => (batch, S, d_model)
            station_features_t = tf.concat(station_embeds, axis=1)

            # Station-level Transformer => (batch, S, d_model)
            spatial_out_t = self.spatial_transformer(station_features_t, training=training)

            # Extract target station (index 0) => (batch, d_model)
            target_emb_t = spatial_out_t[:, 0, :]

            time_embeddings.append(target_emb_t)

        # Stack across time => list of L tensors of shape (batch, d_model)
        # final shape => (batch, L, d_model)
        time_embeddings = tf.stack(time_embeddings, axis=1)

        return time_embeddings


###############################################################################
# 5) BUILD THE FULL MODEL
###############################################################################
def build_spatiotemporal_model(L=3, S=5, num_features=6,
                               d_model=32, num_heads=2, d_ff=64,
                               num_station_transformer_layers=1,
                               num_time_transformer_layers=1):
    """
    Example pipeline:
      1) StationSpatialEncoder => (batch, L, d_model)
      2) TimeTransformer       => (batch, L, d_model)
      3) Take last step or reduce => (batch, d_model)
      4) Dense => (batch, 1)
    """
    main_input = Input(shape=(L, S, num_features), name="main_input")

    # --- Station + Spatial encoding ---
    x = StationSpatialEncoder(
        L=L,
        S=S,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_station_transformer_layers=num_station_transformer_layers
    )(main_input)

    # x shape => (batch, L, d_model)

    # --- Time-level Transformer ---
    x = TimeTransformer(
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_time_transformer_layers
    )(x)

    # If you want just the final time step:
    # shape => (batch, d_model)
    x = x[:, -1, :]

    # Final Dense for regression
    outputs = layers.Dense(1)(x)

    model = Model(inputs=main_input, outputs=outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


###############################################################################
# 6) TRAIN SCRIPT (NO LAMBDAS)
###############################################################################
def main():
    L = 3
    S = 5
    num_features = 6
    d_model = 32
    num_heads = 2
    d_ff = 64

    # Build the model
    model = build_spatiotemporal_model(
        L=L, S=S, num_features=num_features,
        d_model=d_model, num_heads=num_heads, d_ff=d_ff,
        num_station_transformer_layers=1,
        num_time_transformer_layers=1
    )
    model.summary()

    # Dummy data
    num_samples = 100
    X_train = np.random.rand(num_samples, L, S, num_features).astype(np.float32)
    y_train = np.random.rand(num_samples, 1).astype(np.float32)

    # Train
    model.fit(X_train, y_train, epochs=3, batch_size=8, validation_split=0.1)

    # Save in Keras format
    model.save("my_spatiotemporal_model.keras")
    print("Model saved to 'my_spatiotemporal_model.keras'.")


if __name__ == "__main__":
    main()
