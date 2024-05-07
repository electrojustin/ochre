import parmed
import numpy as np
import keras
import tensorflow as tf
import sys
import math
from util import embed_point_cloud
from util import parse_prmtop
from util import write_pdb
from gan import create_generator

def rand_rotation():
  ret = np.random.rand(3, 3)
  ret, _, = np.linalg.qr(ret)
  return ret

def encode_data(coord_filename, atom_types, num_atoms, num_atom_types):
  if coord_filename[-3:] == '.rst' or coord_filename[-4:] == 'rst7':
    coords = parmed.amber.netcdffiles.NetCDFRestart.open_old(netcdf_filename).coordinates
  else:
    coords = parmed.load_file(coord_filename).coordinates
  coords = coords[:1, :num_atoms, :]
  coords[:, :, 0] -= np.average(coords[:, :, 0])
  coords[:, :, 1] -= np.average(coords[:, :, 1])
  coords[:, :, 2] -= np.average(coords[:, :, 2])

  discrete_width = int(math.ceil(num_atoms**(1/3))*2)
  max_search_distance = 5
  latent_dim = 3
  input_coords = np.zeros((1, discrete_width, discrete_width, discrete_width, 3))
  mask = np.zeros((1, discrete_width, discrete_width, discrete_width, 3))
  input_atoms = np.zeros((1, discrete_width, discrete_width, discrete_width))
  indices = embed_point_cloud(coords[0, :, :], discrete_width, max_search_distance)
  for i in range(0, len(indices)):
    index = indices[i]
    input_coords[0, index[0], index[1], index[2], :] = coords[0, i, :]
    mask[0, index[0], index[1], index[2], :] = 1
    input_atoms[0, index[0], index[1], index[2]] = atom_types[i]
  input_coords = tf.convert_to_tensor(input_coords, dtype=tf.float32)
  input_atoms = tf.convert_to_tensor(input_atoms, dtype=tf.int8)
  input_atoms = tf.one_hot(input_atoms - 1, num_atom_types)

  return input_coords, input_atoms, indices, discrete_width, mask

inactive_prmtop = sys.argv[1] + '.prmtop'
inactive_coords = sys.argv[1] + '.inpcrd'
active_prmtop = sys.argv[2] + '.prmtop'
active_coords = sys.argv[2] + '.inpcrd'
num_atoms = int(sys.argv[3])
num_residues = int(sys.argv[4])
dataset_size = int(sys.argv[5])

inactive_atom_types, num_atom_types, inactive_atom_names, _, inactive_res_nums = parse_prmtop(inactive_prmtop, num_atoms)
active_atom_types, _, active_atom_names, _, active_res_nums = parse_prmtop(active_prmtop, num_atoms)

inactive_coords, inactive_atom_types, inactive_gather, discrete_width, inactive_mask = encode_data(inactive_coords, inactive_atom_types, num_atoms, num_atom_types)
active_coords, active_atom_types, active_gather, _, active_mask = encode_data(active_coords, active_atom_types, num_atoms, num_atom_types)


inactive_traj = parmed.amber.netcdffiles.NetCDFRestart.open_old(sys.argv[1] + '.mdcrd').coordinates[:, :num_atoms, :]
print(inactive_traj.shape)
active_traj = parmed.amber.netcdffiles.NetCDFRestart.open_old(sys.argv[2] + '.mdcrd').coordinates[:, :num_atoms, :]

tmp_inactive_res_nums = set()
tmp_active_res_nums = set()
inactive_res_prmtop_indices = {}
active_res_prmtop_indices = {}
inactive_alpha_gather = {}
active_alpha_gather = {}
missing_res_nums = set()
for i in range(0, len(inactive_atom_names)):
  if inactive_atom_names[i].strip() == 'CA':
    if inactive_res_nums[i] < 1:
      continue
    inactive_res_prmtop_indices[inactive_res_nums[i]-1] = i
    inactive_alpha_gather[inactive_res_nums[i]-1] = inactive_gather[i]
    tmp_inactive_res_nums.add(inactive_res_nums[i]-1)
for i in range(0, len(active_atom_names)):
  if active_atom_names[i].strip() == 'CA':
    if active_res_nums[i] < 1:
      continue
    active_res_prmtop_indices[active_res_nums[i]-1] = i
    active_alpha_gather[active_res_nums[i]-1] = active_gather[i]
    tmp_active_res_nums.add(active_res_nums[i]-1)
inactive_res_nums = list(tmp_inactive_res_nums.intersection(tmp_active_res_nums))
inactive_res_nums.sort()
active_res_nums = inactive_res_nums
inactive_alpha_gather = list(map(lambda x: inactive_alpha_gather[x], inactive_res_nums))
active_alpha_gather = list(map(lambda x: active_alpha_gather[x], active_res_nums))

dataset = np.zeros((dataset_size*2, num_residues, 3))
for i in range(0, len(inactive_res_nums)):
  res_num = inactive_res_nums[i]
  alpha_idx = inactive_res_prmtop_indices[res_num]
  dataset[:dataset_size, res_num:res_num+1, :] = inactive_traj[:, alpha_idx:alpha_idx+1, :]
for i in range(0, len(active_res_nums)):
  res_num = active_res_nums[i]
  alpha_idx = active_res_prmtop_indices[res_num]
  dataset[dataset_size:, res_num:res_num+1, :] = active_traj[:, alpha_idx:alpha_idx+1, :]
#dataset[:dataset_size, tmp_inactive_res_nums, :] = inactive_traj
#dataset[dataset_size:, tmp_active_res_Nums, :] = active_traj
for i in range(0, dataset.shape[0]):
  rotation = rand_rotation()
  missing_res_mask = np.zeros(dataset.shape[1:])
  nonzero_data = np.nonzero(dataset[i, :, :])
  missing_res_mask[nonzero_data] = 1
  num_nonzero = np.sum(missing_res_mask)/3
  dataset[i, :, 0] -= np.sum(dataset[i, :, 0])/num_nonzero
  dataset[i, :, 1] -= np.sum(dataset[i, :, 1])/num_nonzero
  dataset[i, :, 2] -= np.sum(dataset[i, :, 2])/num_nonzero
  dataset[i, :, :] *= missing_res_mask
  dataset[i, :, :] = np.dot(rotation, dataset[i, :, :].transpose()).transpose()
dataset = dataset.reshape((dataset_size*2, -1))
dataset = np.append(dataset, np.zeros((2*dataset_size, 1)), axis=1)
dataset[dataset_size:, -1] = 1
np.save('amber_synthetic_train_dataset.npy', dataset)


inactive_res_nums = list(map(lambda x: [x], inactive_res_nums))
active_res_nums = inactive_res_nums


latent_dim = 3
model = create_generator(discrete_width, num_atom_types, num_atoms, latent_dim)
model.compile()
model.load_weights('gen.weights.h5')

seed = 1337
rng = tf.random.Generator.from_seed(seed)

inactive_output_coords = None
print('Generating inactive conformations')
for i in range(0, dataset_size):
  noise = rng.normal(shape=(1, discrete_width, discrete_width, discrete_width, latent_dim), stddev=0.1) * inactive_mask
  output_coords = model(inputs=[inactive_coords, inactive_atom_types, noise])
  if inactive_output_coords is None:
    inactive_output_coords = tf.reshape(tf.scatter_nd(inactive_res_nums, tf.gather_nd(params=output_coords[0, :, :, :, :], indices=inactive_alpha_gather), (167, 3)), (1, 167, 3))
  else:
    output_coords = tf.reshape(tf.scatter_nd(inactive_res_nums, tf.gather_nd(params=output_coords[0, :, :, :, :], indices=inactive_alpha_gather), (167, 3)), (1, 167, 3))
    inactive_output_coords = tf.concat([inactive_output_coords, output_coords], axis=0)
inactive_output_coords = inactive_output_coords.numpy()

active_output_coords = None
print('Generating active conformations')
for i in range(0, dataset_size):
  noise = rng.normal(shape=(1, discrete_width, discrete_width, discrete_width, latent_dim), stddev=0.1) * active_mask
  output_coords = model(inputs=[active_coords, active_atom_types, noise])
  if active_output_coords is None:
    active_output_coords = tf.reshape(tf.scatter_nd(active_res_nums, tf.gather_nd(params=output_coords[0, :, :, :, :], indices=active_alpha_gather), (167, 3)), (1, 167, 3))
  else:
    output_coords = tf.reshape(tf.scatter_nd(active_res_nums, tf.gather_nd(params=output_coords[0, :, :, :, :], indices=active_alpha_gather), (167, 3)), (1, 167, 3))
    active_output_coords = tf.concat([active_output_coords, output_coords], axis=0)
active_output_coords = active_output_coords.numpy()

dataset = np.zeros((dataset_size*2, 167, 3))
dataset[:dataset_size, :, :] = inactive_output_coords
dataset[dataset_size:, :, :] = active_output_coords
for i in range(0, dataset.shape[0]):
  rotation = rand_rotation()
  dataset[i, :, :] = np.dot(rotation, dataset[i, :, :].transpose()).transpose()
dataset = dataset.reshape((dataset_size*2, -1))
dataset = np.append(dataset, np.zeros((2*dataset_size, 1)), axis=1)
dataset[dataset_size:, -1] = 1
np.save('ochre_synthetic_train_dataset.npy', dataset)
