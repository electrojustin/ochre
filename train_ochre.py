import parmed
import math
import numpy as np
import sys
import keras
from keras import layers
import tensorflow as tf
from util import embed_point_cloud
from util import parse_prmtop
from gan import GAN
from gan import BootstrapGenerator

def rand_rotation():
  ret = np.random.rand(3, 3)
  ret, _, = np.linalg.qr(ret)
  return ret

prmtop_filename = sys.argv[1]
netcdf_filename = sys.argv[2]
num_atoms = int(sys.argv[3])

atom_types, num_atom_types, _, _, _ = parse_prmtop(prmtop_filename, num_atoms)

traj = parmed.amber.netcdffiles.NetCDFTraj.open_old(netcdf_filename)
traj = traj.coordinates[:, :num_atoms, :]

discrete_width = int(math.ceil(num_atoms**(1/3))*2)
max_search_distance = 5
temporal_subsample = 1
cycle_size = 2
num_cycles = 1 #int(sys.argv[5]) # Slows down training immensely
num_frames = int(round((traj.shape[0])/temporal_subsample))
#for i in range(0, traj.shape[0]):
#  if not embed_point_cloud(traj[i, :, :], discrete_width, search_distance):
#    print('Error')
input_coords = np.zeros((num_frames, discrete_width, discrete_width, discrete_width, 3))
mask = np.zeros((num_frames, discrete_width, discrete_width, discrete_width, 3))
input_atoms = np.zeros((num_frames, discrete_width, discrete_width, discrete_width))
near_output_coords = np.zeros((num_frames, discrete_width, discrete_width, discrete_width, 3))
far_output_coords = np.zeros((num_frames, discrete_width, discrete_width, discrete_width, 3))
frame_idx = 0
for i in range(num_cycles*cycle_size, traj.shape[0], temporal_subsample):
  flat_inputs = traj[i-num_cycles*cycle_size, :, :]
  flat_near_outputs = traj[i-(num_cycles-1)*cycle_size, :, :]
  flat_far_outputs = traj[i, :, :]
  rotation = rand_rotation()
  flat_inputs[:, 0] -= np.average(flat_inputs[:, 0])
  flat_inputs[:, 1] -= np.average(flat_inputs[:, 1])
  flat_inputs[:, 2] -= np.average(flat_inputs[:, 2])
  flat_near_outputs[:, 0] -= np.average(flat_near_outputs[:, 0])
  flat_near_outputs[:, 1] -= np.average(flat_near_outputs[:, 1])
  flat_near_outputs[:, 2] -= np.average(flat_near_outputs[:, 2])
  flat_far_outputs[:, 0] -= np.average(flat_far_outputs[:, 0])
  flat_far_outputs[:, 1] -= np.average(flat_far_outputs[:, 1])
  flat_far_outputs[:, 2] -= np.average(flat_far_outputs[:, 2])
  rotation = rand_rotation()
  flat_inputs = np.dot(rotation, flat_inputs.transpose()).transpose()
  flat_near_outputs = np.dot(rotation, flat_near_outputs.transpose()).transpose()
  flat_far_outputs = np.dot(rotation, flat_far_outputs.transpose()).transpose()
  #flat_near_outputs -= flat_inputs
  #flat_far_outputs -= flat_inputs
  indices = embed_point_cloud(flat_inputs, discrete_width, max_search_distance)
  for j in range(0, len(indices)):
    index = indices[j]
    input_coords[frame_idx, index[0], index[1], index[2], :] = flat_inputs[j, :]
    mask[frame_idx, index[0], index[1], index[2], :] = 1
    input_atoms[frame_idx, index[0], index[1], index[2]] = atom_types[j]
    near_output_coords[frame_idx, index[0], index[1], index[2], :] = flat_near_outputs[j, :]
    far_output_coords[frame_idx, index[0], index[1], index[2], :] = flat_far_outputs[j, :]
  frame_idx += 1
num_frames = frame_idx
input_coords = input_coords[:num_frames, :, :, :, :]
mask = mask[:num_frames, :, :, :]
input_atoms = input_atoms[:num_frames, :, :, :]
near_output_coords = near_output_coords[:num_frames, :, :, :, :]
far_output_coords = far_output_coords[:num_frames, :, :, :, :]

input_coords = tf.convert_to_tensor(input_coords, dtype=tf.float32)
mask = tf.convert_to_tensor(mask, dtype=tf.float32)
input_atoms = tf.convert_to_tensor(input_atoms, dtype=tf.int8)
near_output_coords = tf.convert_to_tensor(near_output_coords, dtype=tf.float32)
far_output_coords = tf.convert_to_tensor(far_output_coords, dtype=tf.float32)

seed=1337
validation_size = 1
latent_dim = 3
batch_size = 10
dataset = tf.data.Dataset.from_tensor_slices((input_coords, input_atoms, mask, near_output_coords, far_output_coords))
dataset = dataset.shuffle(dataset.cardinality(), seed=seed)
validation_dataset = dataset.take(validation_size).batch(validation_size)
train_dataset = dataset.skip(validation_size).batch(batch_size)

learning_rate = 0.0001
model = GAN(discrete_width, num_atom_types, num_atoms, latent_dim, num_cycles)
restart = False
try:
  model.gen.load_weights('gen.weights.h5')
except:
  print('Error loading generator weights')
  restart = True

try:
    model.near_disc.load_weights('near_disc.weights.h5')
    model.far_disc.load_weights('far_disc.weights.h5')
except:
    print('Error loading discriminator weights')

if restart:
    print('Bootstrapping generator')
    early_stop = keras.callbacks.EarlyStopping(monitor='gen_loss', patience=1, mode='min')
    checkpoint = keras.callbacks.ModelCheckpoint(filepath='bootstrap_checkpoint.keras', monitor='gen_loss', save_best_only=True, mode='min')
    bootstrap = BootstrapGenerator(discrete_width, num_atom_types, num_atoms, latent_dim)
    try:
      bootstrap = keras.models.load_model('bootstrap_checkpoint.keras')
    except:
        print('No bootstrap checkpoint')
    bootstrap.compile(gen_optimizer=keras.optimizers.Adam(learning_rate=0.0001))
    bootstrap.fit(train_dataset, callbacks=[early_stop, checkpoint], epochs=200)

    model.gen = bootstrap.gen

model.compile(near_disc_optimizer=keras.optimizers.Adam(learning_rate=0.001),#learning_rate),
              far_disc_optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              gen_optimizer=keras.optimizers.Adam(learning_rate=learning_rate))

num_epochs = int(sys.argv[4])
checkpoint = keras.callbacks.ModelCheckpoint(filepath='checkpoint.keras', monitor='gen_loss', save_best_only=True, mode='min')
model.fit(train_dataset, epochs=num_epochs, callbacks=[checkpoint])

model.near_disc.save_weights('near_disc.weights.h5')
model.far_disc.save_weights('far_disc.weights.h5')
model.gen.save_weights('gen.weights.h5')

for batch in validation_dataset:
  in_coord, in_atom, mask, _, _ = batch
  noise = model.rng.normal(shape=(in_coord.shape[0], discrete_width, discrete_width, discrete_width, latent_dim), stddev=model.noise_stddev)
  in_atom = tf.one_hot(in_atom - 1, num_atom_types)
  gen_output = model.gen(inputs=[in_coord, in_atom, noise])
  unpack_gen_out = []
  indices = embed_point_cloud(traj[0, :, :], discrete_width, max_search_distance)
  for index in indices:
    unpack_gen_out.append(gen_output[0, index[0], index[1], index[2], :])
  print(np.array(unpack_gen_out))
