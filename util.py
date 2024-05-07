import numpy as np
import parmed
import math

amber_type_idx = {}
with open('parm14ipq.dat', 'r') as in_file:
  in_file.readline()
  idx = 1 # reserve 0
  line = in_file.readline()
  while line.strip():
    amber_type_idx[line.split(' ')[0]] = idx
    idx += 1
    line = in_file.readline()

def parse_prmtop(prmtop_filename, num_atoms):
  top = parmed.load_file(prmtop_filename)
  num_atom_types = len(list(amber_type_idx.keys()))
  atom_types = list(map(lambda x: amber_type_idx[x.type], top.atoms[:num_atoms]))
  atom_names = list(map(lambda x: x.name, top.atoms[:num_atoms]))
  residue_names = list(map(lambda x: x.residue.name, top.atoms[:num_atoms]))
  residue_nums = list(map(lambda x: x.residue.number, top.atoms[:num_atoms]))
  
  return atom_types, num_atom_types, atom_names, residue_names, residue_nums

def write_pdb(out_filename, atom_names, residue_names, residue_nums, coords):
  with open(out_filename, 'w') as out_file:
    for i in range(0, len(atom_names)):
      x = '{:.3f}'.format(coords[i][0])
      y = '{:.3f}'.format(coords[i][1])
      z = '{:.3f}'.format(coords[i][2])
      line = 'ATOM   ' + str(i+1).rjust(4) + ' '
      line += atom_names[i].ljust(3).rjust(4) + ' '
      line += residue_names[i] + '  '
      line += str(residue_nums[i]).rjust(4) + ' '*4
      line += x.rjust(8) + y.rjust(8) + z.rjust(8) + ' '*23
      line += atom_names[i][0] + '  \n'
      out_file.write(line)

def embed_point_cloud(points, size, max_search_distance):
  ret = []
  cells = np.zeros((size, size, size))
  x_mapping = {}
  y_mapping = {}
  z_mapping = {}
  by_x = np.argsort(points[:, 0])
  by_y = np.argsort(points[:, 1])
  by_z = np.argsort(points[:, 2])
  bin_num = 0
  start = 0
  bin_size = int(math.ceil(points.shape[0] / size))
  end = bin_size
  while start < points.shape[0]:
    for k in range(start, min(end, points.shape[0])):
      x_mapping[by_x[k]] = bin_num
      y_mapping[by_y[k]] = bin_num
      z_mapping[by_z[k]] = bin_num
    bin_num += 1
    start = end
    end += bin_size
  for j in range(0, points.shape[0]):
    x = x_mapping[j]
    y = y_mapping[j]
    z = z_mapping[j]
    found_spot = False
    if not cells[x, y, z]:
      cells[x, y, z] = 1
      ret.append([x, y, z])
      continue
    for search_distance in range(1, max_search_distance+1):
      for x_delta in range(max(x-search_distance, 0), min(x+search_distance+1, cells.shape[0])):
        for y_delta in range(max(y-search_distance, 0), min(y+search_distance+1, cells.shape[1])):
          for z_delta in range(max(z-search_distance, 0), min(z+search_distance+1, cells.shape[2])):
            if not cells[x_delta, y_delta, z_delta]:
              cells[x_delta, y_delta, z_delta] = 1
              found_spot = True
              ret.append([x_delta, y_delta, z_delta])
              break
          if found_spot:
            break
        if found_spot:
          break
      if found_spot:
        break
    if not found_spot:
      return None

  return ret
