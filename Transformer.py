from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
import tensorflow_addons as tfa


class Patches(layers.Layer):
    def __init__(self, patch_size, overlap_fraction):
        super().__init__()
        self.patch_size = patch_size
        self.overlap_fraction = overlap_fraction

    def call(self, data):
        batch_size = tf.shape(data)[0]
        stride = int(self.patch_size * (1 - self.overlap_fraction))
        patches = tf.image.extract_patches(
            images=data,
            sizes=[1, 1, self.patch_size, 1],
            strides=[1, 1, stride, 1],
            rates=[1, 1, 1, 1],
            padding="SAME",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

from keras import backend as K
from keras.layers import Layer, Dense, Concatenate, Reshape

class FeatureLayer(keras.layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(FeatureLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.dense_layers = []
        for i in range(input_shape[1]):
            self.dense_layers.append(layers.Dense(5, use_bias=True,
                                                  kernel_initializer='random_normal',
                                                  bias_initializer='zeros'))
        super(FeatureLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        outputs = []
        for i, dense_layer in enumerate(self.dense_layers):
            x_slice = x[:, i, :]
            x_slice = dense_layer(x_slice)
            outputs.append(x_slice)
        output = layers.Concatenate()(outputs)
        output = layers.Reshape((x.shape[1], self.output_dim))(output)
        return output  
        
        '''
        The part beneth this is not executed. This part is written for future version as a
        to do work to map parallaly instead of loop. 
        '''

        def apply_dense(x_slice):
            return dense_layer(x_slice)
        outputs = K.map_fn(apply_dense, x)
        output = Concatenate()(outputs)
        output = Reshape((x.shape[1], self.output_dim))(output)
        return output


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.output_dim)
    
    def get_config(self):
        return {"output_dim": self.output_dim}

    

class DotProductAttention(layers.Layer):
    def __init__(self, key_dim, num_heads, dropout=0.0):
        super().__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.wq = layers.Dense(key_dim)
        self.wk = layers.Dense(key_dim)
        self.wv = layers.Dense(key_dim)
        self.dropout_layer = layers.Dropout(dropout)
    
    def call(self, query, key, value, mask=None):
        q = self.wq(query)
        k = self.wk(key)
        v = self.wv(value)
        
        # Scale the dot product
        q = q / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        
        # Calculate dot product attention
        dot_product = tf.matmul(q, k, transpose_b=True)
        
        # Apply mask and softmax
        if mask is not None:
            dot_product = dot_product + (mask * -1e9)
        attention_weights = tf.nn.softmax(dot_product, axis=-1)
        
        # Apply dropout
        attention_weights = self.dropout_layer(attention_weights)
        
        # Calculate the context vector
        context_vector = tf.matmul(attention_weights, v)
        
        return context_vector, attention_weights

class MultiHeadAttention(layers.Layer):
    def __init__(self, key_dim, num_heads, dropout):
        super().__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.query_dense = layers.Dense(key_dim, use_bias=False)
        self.key_dense = layers.Dense(key_dim, use_bias=False)
        self.value_dense = layers.Dense(key_dim, use_bias=False)
        self.dot_product_attention = DotProductAttention()
        self.concatenation = layers.Concatenate()
        self.output_dense = layers.Dense(key_dim, use_bias=False)

    def call(self, inputs):
        queries, keys, values = inputs
        queries = self.query_dense(queries)
        keys = self.key_dense(keys)
        values = self.value_dense(values)

        # splitting the queries, keys and values into heads
        queries = tf.concat(tf.split(queries, self.num_heads, axis=-1), axis=0)
        keys = tf.concat(tf.split(keys, self.num_heads, axis=-1), axis=0)
        values = tf.concat(tf.split(values, self.num_heads, axis=-1), axis=0)

        # apply dot product attention
        attention = self.dot_product_attention([queries, keys, values])
        attention = tf.transpose(attention, perm=[1, 0, 2])
        attention = tf.nn.dropout(attention, rate=self.dropout)

        # concatenate the attention for each head
        concat_attention = self.concatenation([attention])

        # project the concatenated attention to the final output size
        output = self.output_dense(concat_attention)
        return output

class TransformerEncoder(layers.Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout = dropout
    
    def build(self, input_shape):
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = MultiHeadAttention(key_dim=self.head_size, num_heads=self.num_heads, dropout=self.dropout)
        self.dropout1 = layers.Dropout(self.dropout)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.ff = Sequential([
            layers.Conv1D(filters=self.ff_dim, kernel_size=1, activation="relu"),
            layers.Dropout(self.dropout),
            layers.Conv1D(filters=input_shape[-1], kernel_size=1)
        ])
        super(TransformerEncoder, self).build(input_shape)
        
    def call(self, inputs, training=None):
        x = self.norm1(inputs)
        x = self.attn(x, x)
        x = self.dropout1(x)
        x = self.norm2(x + inputs)
        x = self.ff(x)
        return x + inputs

def build_model(input_shape, head_size, num_heads, ff_dim, num_pyramidal_blocks, mlp_units,patch_size,num_patches,overlap_fraction, dropout=0, mlp_dropout=0):
    inputs = keras.Input(shape=input_shape)

    # Create patches.
    patches = Patches(patch_size, overlap_fraction)(inputs)
    # Encode patches
    encoded_patches = PatchEncoder(num_patches = num_patches, projection_dim = head_size)(patches)
    x = encoded_patches

    for _ in range(num_pyramidal_blocks):
        #x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
        x = TransformerEncoder(head_size, num_heads, ff_dim, dropout)(x)
        x = layers.Conv1D(filters=40, kernel_size=16, strides = 2, activation="selu")(x)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)

    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(1, activation="linear")(x)
    return keras.Model(inputs, outputs)

if __name__ == '__main__':
    input_shape = (1, 4600, 1)
    patch_size = 8
    data_size = 4600
    overlap_fraction = 0.
    num_patches = int(data_size// (patch_size * (1 - overlap_fraction)))
    print(num_patches)

    # Build the model
    model = build_model(input_shape, head_size=256, num_heads=4, ff_dim=4, num_pyramidal_blocks=3,
                        mlp_units=[128,32,16],patch_size = patch_size,num_patches = num_patches,overlap_fraction = overlap_fraction, mlp_dropout=0.4, dropout=0.25)

    # Compile and train the model
    learning_rate = 0.001
    weight_decay = 0.0001
    model.compile(
                loss="mse",
                optimizer=tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
                metrics=["mae"],
                )
    model.summary()

