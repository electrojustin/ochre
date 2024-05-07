import parmed
import math
import numpy as np
import sys
import keras
import time
from keras import layers
import tensorflow as tf
from util import embed_point_cloud
from util import parse_prmtop
from util import write_pdb
from gan import create_generator

prmtop_filename = sys.argv[1]
coord_filename = sys.argv[2]
num_atoms = int(sys.argv[3])

atom_types, num_atom_types, atom_names, residue_names, residue_nums = parse_prmtop(prmtop_filename, num_atoms)

if coord_filename[-3:] == '.rst' or coord_filename[-4:] == 'rst7':
  coords = parmed.amber.netcdffiles.NetCDFRestart.open_old(netcdf_filename).coordinates
else:
  coords = parmed.load_file(coord_filename).coordinates
coords = coords[:1, :num_atoms, :]
print(coords.shape)

coords[:, :, 0] -= np.average(coords[:, :, 0])
coords[:, :, 1] -= np.average(coords[:, :, 1])
coords[:, :, 2] -= np.average(coords[:, :, 2])


write_pdb('orig.pdb', atom_names, residue_names, residue_nums, coords[0, :, :])

discrete_width = int(math.ceil(num_atoms**(1/3))*2)
max_search_distance = 5
latent_dim = 3
input_coords = np.zeros((1, discrete_width, discrete_width, discrete_width, 3))
mask = np.zeros((1, discrete_width, discrete_width, discrete_width, 3))
input_atoms = np.zeros((1, discrete_width, discrete_width, discrete_width))
indices = embed_point_cloud(coords[0, :, :], discrete_width, max_search_distance)
#print(atom_types)
for i in range(0, len(indices)):
  index = indices[i]
  input_coords[0, index[0], index[1], index[2], :] = coords[0, i, :]
  mask[0, index[0], index[1], index[2], :] = 1
  input_atoms[0, index[0], index[1], index[2]] = atom_types[i]
input_coords = tf.convert_to_tensor(input_coords, dtype=tf.float32)
input_atoms = tf.convert_to_tensor(input_atoms, dtype=tf.int8)
input_atoms = tf.one_hot(input_atoms - 1, num_atom_types)
print(tf.reduce_sum(input_atoms))

model = create_generator(discrete_width, num_atom_types, num_atoms, latent_dim)
model.compile()
model.load_weights('gen.weights.h5')

seed = 1337
num_cycles = int(sys.argv[4])
rng = tf.random.Generator.from_seed(seed)
#out_file = parmed.amber.netcdffiles.NetCDFTraj.open_new('out.nc', num_atoms, False)
#out_coords = np.zeros((num_atoms, 3))
start_time = time.time()
for i in range(0, num_cycles):
  print('Cycle ' + str(i))

#  for j in range(0, len(indices)):
#    index = indices[j]
#    out_coords[j, :] = input_coords[0, index[0], index[1], index[2], :]
#  out_file.add_coordinates(out_coords.copy())

  noise = rng.normal(shape=(1, discrete_width, discrete_width, discrete_width, latent_dim), stddev=0.1) * mask
  input_coords = model(inputs=[input_coords, input_atoms, noise]) * mask 

print('Time! ' + str(time.time() - start_time))
  
#out_coords = []
#for i in range(0, len(indices)):
#  index = indices[i]
#  out_coords.append(input_coords[0, index[0], index[1], index[2], :])
out_coords = tf.gather_nd(params=input_coords[0, :, :, :, :], indices=indices)
write_pdb('new.pdb', atom_names, residue_names, residue_nums, out_coords)
