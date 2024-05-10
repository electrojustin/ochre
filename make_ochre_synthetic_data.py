import parmed
import numpy as np
import keras
import tensorflow as tf
import sys
import math
from util import embed_point_cloud
from util import parse_prmtop
from util import write_pdb
from util import rand_rotation
from gan import create_generator

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

prmtop_filename = sys.argv[1]
coords_filename = sys.argv[2]
num_atoms = int(sys.argv[3])
pdb = sys.argv[4]
dataset_size = int(sys.argv[5])
label = int(sys.argv[6])

atom_types, num_atom_types, atom_names, _, _ = parse_prmtop(prmtop_filename, num_atoms)

coords, atom_types, gather_indices, discrete_width, mask = encode_data(coords_filename, atom_types, num_atoms, num_atom_types)

alpha_gather_indices = []
# Filter gather indices for alpha carbons
for i in range(0, len(atom_names)):
  if atom_names[i] == 'CA':
    alpha_gather_indices.append(gather_indices[i])

latent_dim = 3
model = create_generator(discrete_width, num_atom_types, num_atoms, latent_dim)
model.compile()
model.load_weights('gen.weights.h5')

seed = 1337
rng = tf.random.Generator.from_seed(seed)

output_coords = None
print('Generating conformations')
for i in range(0, dataset_size):
  noise = rng.normal(shape=(1, discrete_width, discrete_width, discrete_width, latent_dim), stddev=0.1) * mask
  tmp = model(inputs=[coords, atom_types, noise])
  if output_coords is None:
    output_coords = tf.reshape(tf.gather_nd(params=tmp[0, :, :, :, :], indices=alpha_gather_indices), (1, len(alpha_gather_indices), 3))
  else:
    tmp = tf.reshape(tf.gather_nd(params=tmp[0, :, :, :, :], indices=alpha_gather_indices), (1, len(alpha_gather_indices), 3))
    output_coords = tf.concat([output_coords, tmp], axis=0)
output_coords = output_coords.numpy()

with open('kras_dataset/alignment.csv', 'r') as align_file:
  for row in align_file.readlines():
    if pdb+'\tA' in row:
      parsed_row = row.split('\t')
      indices = list(map(lambda x: int(x), parsed_row[-1].split(',')))

output_coords = output_coords[:, indices, :]

dataset = np.zeros((dataset_size, output_coords.shape[1]*3+1))
dataset[:, -1] = label
for i in range(0, output_coords.shape[0]):
  output_coords[i, :, 0] -= np.average(output_coords[i, :, 0])
  output_coords[i, :, 1] -= np.average(output_coords[i, :, 1])
  output_coords[i, :, 2] -= np.average(output_coords[i, :, 2])
  rotation = rand_rotation()
  dataset[i, :-1] = np.dot(rotation, output_coords[i, :, :].transpose()).transpose().flatten()

np.save(sys.argv[7], dataset)
