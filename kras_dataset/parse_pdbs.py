import numpy as np

dataset = []
num_residues = 167

with open('kras_data.csv', 'r') as manifest_file:
  for row in manifest_file.readlines():
    alpha_carbons = np.zeros((num_residues, 3))
    parsed_row = row.strip().split('\t')
    pdb = parsed_row[0]
    chain = parsed_row[1]
    label = float(parsed_row[2])
    with open('pdbs/' + pdb + '.pdb', 'r') as pdb_file:
      for row in pdb_file.readlines():
        if row[:4] != 'ATOM':
          continue
        if row[21] != chain:
          continue
        if row[12:16].strip() != 'CA':
          continue
        residue = int(row[22:26].strip())-1
        if residue < 1 or residue > num_residues:
          continue
        x = float(row[30:38].strip())
        y = float(row[38:46].strip())
        z = float(row[46:54].strip())
        alpha_carbons[residue-1, 0] = x
        alpha_carbons[residue-1, 1] = y
        alpha_carbons[residue-1, 2] = z

    alpha_carbons[:, 0] -= np.average(alpha_carbons[:, 0])
    alpha_carbons[:, 1] -= np.average(alpha_carbons[:, 1])
    alpha_carbons[:, 2] -= np.average(alpha_carbons[:, 2])
    datapoint = list(alpha_carbons.flatten())
    datapoint.append(label)
    dataset.append(datapoint)

dataset = np.array(dataset)
np.save('test_dataset.npy', dataset)
