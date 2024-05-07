import numpy as np
import keras
from keras import layers
import tensorflow as tf

seed = 1337
disc_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.01, seed=seed)
gen_initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.001, seed=seed)
filter_shape = (5, 5, 5)
drop_rate = 0.2

def create_discriminator(discrete_width, num_atom_types, num_atoms, latent_dim):
  vocab = list(range(0, num_atom_types))
  coord_in = layers.Input((discrete_width, discrete_width, discrete_width, 3))
  atom_in = layers.Input((discrete_width, discrete_width, discrete_width, num_atom_types))
  coord_out = layers.Input((discrete_width, discrete_width, discrete_width, 3))  
  x = layers.Concatenate(axis=-1)([coord_in, atom_in, coord_out])
  x = layers.Conv3D(64, filter_shape, data_format='channels_last', padding='same', use_bias=True, kernel_initializer=disc_initializer, bias_initializer=disc_initializer)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(drop_rate)(x)
  x = layers.Conv3D(32, filter_shape, data_format='channels_last', padding='same', use_bias=True, kernel_initializer=disc_initializer, bias_initializer=disc_initializer)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(drop_rate)(x)
  x = layers.Conv3D(32, filter_shape, data_format='channels_last', padding='same', use_bias=True, kernel_initializer=disc_initializer, bias_initializer=disc_initializer)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(drop_rate)(x)
  x = layers.Conv3D(16, filter_shape, data_format='channels_last', padding='same', use_bias=True, kernel_initializer=disc_initializer, bias_initializer=disc_initializer)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(drop_rate)(x)
  x = layers.Conv3D(16, filter_shape, data_format='channels_last', padding='same', use_bias=True, kernel_initializer=disc_initializer, bias_initializer=disc_initializer)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(drop_rate)(x)
  x = layers.Conv3D(8, filter_shape, data_format='channels_last', padding='same', use_bias=True, kernel_initializer=disc_initializer, bias_initializer=disc_initializer)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(drop_rate)(x)
  x = layers.Conv3D(4, filter_shape, data_format='channels_last', padding='same', use_bias=True, kernel_initializer=disc_initializer, bias_initializer=disc_initializer)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(drop_rate)(x)
  x = layers.Conv3D(1, filter_shape, data_format='channels_last', padding='same', use_bias=True, activation='tanh', kernel_initializer=disc_initializer, bias_initializer=disc_initializer)(x)
  x = layers.Flatten()(x)
  x = layers.Reshape((-1, 1))(x)
  x = layers.GlobalAveragePooling1D(data_format='channels_last')(x)
  return keras.Model(inputs=[coord_in, atom_in, coord_out], outputs=x)

def create_generator(discrete_width, num_atom_types, num_atoms, latent_dim):
  vocab = list(range(0, num_atom_types))
  coord_in = layers.Input((discrete_width, discrete_width, discrete_width, 3))
  atom_in = layers.Input((discrete_width, discrete_width, discrete_width, num_atom_types))
  noise = layers.Input((discrete_width, discrete_width, discrete_width, 3))
  x = layers.Add()([coord_in, noise])
  x = layers.Concatenate(axis=-1)([x, atom_in])
  x = layers.Conv3D(64, filter_shape, data_format='channels_last', padding='same', use_bias=True, kernel_initializer=gen_initializer, bias_initializer=gen_initializer)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(drop_rate)(x)
  x = layers.Conv3D(32, filter_shape, data_format='channels_last', padding='same', use_bias=True, kernel_initializer=gen_initializer, bias_initializer=gen_initializer)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(drop_rate)(x)
  x = layers.Conv3D(32, filter_shape, data_format='channels_last', padding='same', use_bias=True, kernel_initializer=gen_initializer, bias_initializer=gen_initializer)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(drop_rate)(x)
  x = layers.Conv3D(16, filter_shape, data_format='channels_last', padding='same', use_bias=True, kernel_initializer=gen_initializer, bias_initializer=gen_initializer)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(drop_rate)(x)
  x = layers.Conv3D(16, filter_shape, data_format='channels_last', padding='same', use_bias=True, kernel_initializer=gen_initializer, bias_initializer=gen_initializer)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(drop_rate)(x)
  x = layers.Conv3D(8, filter_shape, data_format='channels_last', padding='same', use_bias=True, kernel_initializer=gen_initializer, bias_initializer=gen_initializer)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(drop_rate)(x)
  x = layers.Conv3D(4, filter_shape, data_format='channels_last', padding='same', use_bias=True, kernel_initializer=gen_initializer, bias_initializer=gen_initializer)(x)
  x = layers.LeakyReLU()(x)
  x = layers.Dropout(drop_rate)(x)
  x = layers.Conv3D(3, filter_shape, data_format='channels_last', padding='same', use_bias=True, kernel_initializer=gen_initializer, bias_initializer=gen_initializer)(x)
  return keras.Model(inputs=[coord_in, atom_in, noise], outputs=x)

class BootstrapGenerator(keras.Model):
  def __init__(self, discrete_width, num_atom_types, num_atoms, latent_dim, gen=None, **kwargs):
    super().__init__(**kwargs)
    self.discrete_width = discrete_width
    self.latent_dim = latent_dim
    self.num_atom_types = num_atom_types
    self.num_atoms = num_atoms
    self.gen_loss_tracker = keras.metrics.Mean(name='gen_loss')
    if gen:
      self.gen = gen
    else:
      self.gen = create_generator(discrete_width, num_atom_types, num_atoms, latent_dim)
    self.rng = tf.random.Generator.from_seed(seed)
    self.noise_stddev = 0.1

  @property
  def metrics(self):
    return [self.gen_loss_tracker]

  def compile(self, gen_optimizer):
    super().compile()
    self.gen_optimizer = gen_optimizer

  def train_step(self, data):
    real_in_coords, real_in_atoms, mask, real_out_near_coords, _ = data
    batch_size = tf.shape(real_in_coords)[0]
    real_in_atoms = tf.one_hot(real_in_atoms - 1, self.num_atom_types)

    with tf.GradientTape() as tape:
      real_log_var = tf.math.log(tf.math.reduce_variance(real_out_near_coords - real_in_coords))
      noise = self.rng.normal(shape=(batch_size, self.discrete_width, self.discrete_width, self.discrete_width, self.latent_dim), stddev=self.noise_stddev) * mask
      gen_output = self.gen(inputs=[real_in_coords, real_in_atoms, noise])
      gen_log_var = tf.math.log(tf.math.reduce_variance(gen_output - real_in_coords))
      gen_loss = tf.math.reduce_sum((real_log_var - gen_log_var)**2)
    grads = tape.gradient(gen_loss, self.gen.trainable_weights)
    self.gen_optimizer.apply_gradients(zip(grads, self.gen.trainable_weights))

    self.gen_loss_tracker.update_state(gen_loss)
    return {
      'gen_loss': self.gen_loss_tracker.result(),
      'log_real_var': real_log_var,
      'log_gen_var': gen_log_var,
    }

  def get_config(self):
    config = super().get_config()
    config.update(
        {
            'gen': self.gen,
            'discrete_width': self.discrete_width,
            'num_atom_types': self.num_atom_types,
            'num_atoms': self.num_atoms,
            'latent_dim': self.latent_dim,
        })
    return config

  @classmethod
  def from_config(cls, config):
    config['gen'] = keras.layers.deserialize(config['gen'])
    return cls(**config)


class GAN(keras.Model):
  def __init__(self, discrete_width, num_atom_types, num_atoms, latent_dim, num_cycles, near_disc=None, far_disc=None, gen=None, **kwargs):
    super().__init__(**kwargs)
    self.discrete_width = discrete_width
    self.latent_dim = latent_dim
    self.num_atom_types = num_atom_types
    self.num_cycles = num_cycles
    self.num_atoms = num_atoms
    self.gen_loss_tracker = keras.metrics.Mean(name='gen_loss')
    self.near_disc_loss_tracker = keras.metrics.Mean(name='near_disc_loss')
    self.far_disc_loss_tracker = keras.metrics.Mean(name='far_disc_loss')

    self.rng = tf.random.Generator.from_seed(seed)

    if near_disc:
      self.near_disc = near_disc
    else:
      self.near_disc = create_discriminator(discrete_width, num_atom_types, num_atoms, latent_dim)
    if far_disc:
      self.far_disc = near_disc
    else:
      self.far_disc = create_discriminator(discrete_width, num_atom_types, num_atoms, latent_dim)
    if gen:
      self.gen = gen
    else:
      self.gen = create_generator(discrete_width, num_atom_types, num_atoms, latent_dim)
    self.noise_stddev = 0.1
    self.var_loss_weight = 0.1

  @property
  def metrics(self):
    return [self.gen_loss_tracker, self.near_disc_loss_tracker, self.far_disc_loss_tracker]

  def compile(self, near_disc_optimizer, far_disc_optimizer, gen_optimizer):
    super().compile()
    self.near_disc_optimizer = near_disc_optimizer
    self.far_disc_optimizer = far_disc_optimizer
    self.gen_optimizer = gen_optimizer

  def train_step(self, data):
    real_in_coords, real_in_atoms, mask, real_out_near_coords, real_out_far_coords = data
    batch_size = tf.shape(real_in_coords)[0]
    real_in_atoms = tf.one_hot(real_in_atoms - 1, self.num_atom_types)

    # Generate random conformations
    in_coords = real_in_coords
    for i in range(0, self.num_cycles):
      noise = self.rng.normal(shape=(batch_size, self.discrete_width, self.discrete_width, self.discrete_width, self.latent_dim), stddev=self.noise_stddev) * mask
      gen_output = self.gen(inputs=[in_coords,
                                    real_in_atoms,
                                    noise])
      in_coords = gen_output * mask
      if i == 0:
        gen_near_output = tf.identity(in_coords)
    gen_far_output = in_coords

    # Train near discriminator
    with tf.GradientTape() as tape:
      real_pred = self.near_disc(inputs=[real_in_coords, real_in_atoms, real_out_near_coords])
      fake_pred = self.near_disc(inputs=[real_in_coords, real_in_atoms, gen_near_output])
      # Wassenstein discriminator loss
      near_disc_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
    grads = tape.gradient(near_disc_loss, self.near_disc.trainable_weights)
    self.near_disc_optimizer.apply_gradients(zip(grads, self.near_disc.trainable_weights))

    # Train far discriminator
#    with tf.GradientTape() as tape:
#      real_pred = self.far_disc(inputs=[real_in_coords, real_in_atoms, real_out_far_coords])
#      fake_pred = self.far_disc(inputs=[real_in_coords, real_in_atoms, gen_far_output])
      # Wassenstein discriminator loss
#      far_disc_loss = tf.reduce_mean(fake_pred) - tf.reduce_mean(real_pred)
#    grads = tape.gradient(far_disc_loss, self.far_disc.trainable_weights)
#    self.far_disc_optimizer.apply_gradients(zip(grads, self.far_disc.trainable_weights))

    # Train generator
    with tf.GradientTape() as tape:
      real_log_var = tf.math.log(tf.math.reduce_variance(real_out_near_coords - real_in_coords))

      in_coords = real_in_coords
      for i in range(0, self.num_cycles):
        noise = self.rng.normal(shape=(batch_size, self.discrete_width, self.discrete_width, self.discrete_width, self.latent_dim), stddev=self.noise_stddev) * mask
        gen_output = self.gen(inputs=[in_coords, real_in_atoms, noise])
        in_coords = gen_output * mask
        if i == 0:
          gen_near_output = tf.identity(in_coords)
      gen_far_output = in_coords

      gen_log_var = tf.math.log(tf.math.reduce_variance(gen_near_output - real_in_coords))
      near_pred = self.near_disc(inputs=[real_in_coords, real_in_atoms, gen_near_output])
#      far_pred = self.far_disc(inputs=[real_in_coords, real_in_atoms, gen_far_output])

      var_loss = tf.math.reduce_sum((real_log_var - gen_log_var)**2)

      gen_loss = -tf.reduce_mean(near_pred) + var_loss * self.var_loss_weight
#      gen_loss = -tf.reduce_mean(near_pred)-tf.reduce_mean(far_pred) + var_loss * self.var_loss_weight
    grads = tape.gradient(gen_loss, self.gen.trainable_weights)
    self.gen_optimizer.apply_gradients(zip(grads, self.gen.trainable_weights))

    self.gen_loss_tracker.update_state(gen_loss)
    self.near_disc_loss_tracker.update_state(near_disc_loss)
#    self.far_disc_loss_tracker.update_state(far_disc_loss)
    return {
      'gen_var': var_loss,
      'gen_loss': self.gen_loss_tracker.result(),
      'near_disc_loss': self.near_disc_loss_tracker.result(),
#      'far_disc_loss': self.far_disc_loss_tracker.result(),
    }

  def get_config(self):
    config = super().get_config()
    config.update(
        {
            'gen': self.gen,
            'near_disc': self.near_disc,
            'far_disc': self.far_disc,
            'discrete_width': self.discrete_width,
            'num_atom_types': self.num_atom_types,
            'num_atoms': self.num_atoms,
            'latent_dim': self.latent_dim,
            'num_cycles': self.num_cycles,
        })
    return config

  @classmethod
  def from_config(cls, config):
    config['gen'] = keras.layers.deserialize(config['gen'])
    config['near_disc'] = keras.layers.deserialize(config['near_disc'])
    config['far_disc'] = keras.layers.deserialize(config['far_disc'])
    return cls(**config)
