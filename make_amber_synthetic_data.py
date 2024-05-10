import sys
import parmed
import numpy as np
from util import parse_prmtop
from util import rand_rotation

prmtop_filename = sys.argv[1]
traj_filename = sys.argv[2]
num_atoms = int(sys.argv[3])
pdb = sys.argv[4]
dataset_size = int(sys.argv[5])
label = int(sys.argv[6])

atom_types, num_atom_types, atom_names, _, _ = parse_prmtop(prmtop_filename, num_atoms)
traj = parmed.amber.netcdffiles.NetCDFRestart.open_old(traj_filename).coordinates[:, :num_atoms, :]

alpha_gather_indices = []
for i in range(0, len(atom_names)):
  if atom_names[i] == 'CA':
    alpha_gather_indices.append(i)

output_coords = traj[:, alpha_gather_indices, :]

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
