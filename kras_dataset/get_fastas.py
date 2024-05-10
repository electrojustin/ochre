import difflib
import os
import numpy as np

consensus = None

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

sequences = []
bad_seq = 0
with open('kras_data.csv', 'r') as manifest_file:
  for row in manifest_file.readlines():
    parsed_row = row.split('\t')
    pdb = parsed_row[0]
    chain = parsed_row[1]
    seq = ''
    with open('pdbs/' + pdb + '.pdb') as pdb_file:
      for pdb_row in pdb_file.readlines():
        if pdb_row[:4] != 'ATOM':
          continue
        if pdb_row[12:16].strip() != 'CA':
          continue
        if pdb_row[21] != chain:
          continue
        res = pdb_row[17:20]
        seq += amino_sym[res]

    print('>' + pdb + '_' + chain)
    print(seq[:200])
