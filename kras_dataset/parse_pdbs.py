import numpy as np

dataset = []

amino_sym = {
  'ALA': 'A',
  'ARG': 'R',
  'ASN': 'N',
  'ASP': 'D',
  'CYS': 'C',
  'GLN': 'Q',
  'GLU': 'E',
  'GLY': 'G',
  'HIS': 'H',
  'ILE': 'I',
  'LEU': 'L',
  'LYS': 'K',
  'MET': 'M',
  'PHE': 'F',
  'PRO': 'P',
  'SER': 'S',
  'THR': 'T',
  'TRP': 'W',
  'TYR': 'Y',
  'VAL': 'V',
}

alignments = {}
with open('alignment.csv', 'r') as align_file:
  for row in align_file.readlines():
    if 'Consensus' in row:
      continue
    parsed_row = row.split('\t')
    alignments[(parsed_row[0], parsed_row[1])] = list(map(lambda x: int(x), parsed_row[-1].split(',')))

with open('kras_data.csv', 'r') as manifest_file:
  for row in manifest_file.readlines():
    alpha_carbons = []
    parsed_row = row.strip().split('\t')
    pdb = parsed_row[0]
    chain = parsed_row[1]
    label = float(parsed_row[2])
    seq = ''
    with open('pdbs/' + pdb + '.pdb', 'r') as pdb_file:
      for row in pdb_file.readlines():
        if row[:4] != 'ATOM':
          continue
        if row[21] != chain:
          continue
        if row[12:16].strip() != 'CA':
          continue
        seq += amino_sym[row[17:20]]
        x = float(row[30:38].strip())
        y = float(row[38:46].strip())
        z = float(row[46:54].strip())
        alpha_carbons.append([x, y, z])

    consensus = ''
    for index in alignments[(pdb, chain)]:
      consensus += seq[index]
    alpha_carbons = np.array(alpha_carbons)[alignments[(pdb, chain)]]
    print(consensus)
    alpha_carbons[:, 0] -= np.average(alpha_carbons[:, 0])
    alpha_carbons[:, 1] -= np.average(alpha_carbons[:, 1])
    alpha_carbons[:, 2] -= np.average(alpha_carbons[:, 2])
    datapoint = list(alpha_carbons.flatten())
    datapoint.append(label)
    dataset.append(datapoint)
    print(pdb + ' ' + str(label))

dataset = np.array(dataset)
np.save('test_dataset.npy', dataset)
