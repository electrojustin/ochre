import os

pdbs = set()
with open('kras_data.csv', 'r') as in_file:
  for row in in_file.readlines():
    pdb = row.split('\t')[0]
    pdbs.add(pdb)

os.chdir('pdbs')
for pdb in pdbs:
  if os.path.isfile(pdb + '.pdb'):
    continue
  os.system('wget http://files.rcsb.org/download/' + pdb + '.pdb')
  os.system('wget http://www.rcsb.org/fasta/entry/' + pdb)
  os.rename(pdb, pdb + '.FASTA')
