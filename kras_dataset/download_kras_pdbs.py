import os

pdbs = set()
with open('kras_data.csv', 'r') as in_file:
  for row in in_file.readlines():
    pdb = row.split('\t')[0]
    pdbs.add(pdb)

os.chdir('pdbs')
for pdb in pdbs:
  os.system('wget https://files.rcsb.org/download/' + pdb + '.pdb')
