seqs = {}
indices = {}
out_indices = {}
max_index = 200
with open('msa.txt', 'r') as msa_file:
  for row in msa_file.readlines():
    if '_' in row:
      pdb = row[:4]
      chain = row[5]
      seq = row[12:]
      if '\t' in seq:
        seq = seq[:seq.find('\t')]
      if (pdb, chain) not in seqs:
        seqs[(pdb, chain)] = ''
        indices[(pdb, chain)] = 0
        out_indices[(pdb, chain)] = []
      seqs[(pdb, chain)] += seq

consensus = ''
for i in range(0, max_index):
  first_key = ('6CU6', 'A')
  first_residue = seqs[first_key][i]
  should_output = True
  for key in seqs.keys():
    if seqs[key][i] != first_residue:
      should_output = False
    if seqs[key][i] != '-':
      indices[key] += 1
  if should_output:
    for key in seqs.keys():
      out_indices[key].append(indices[key]-1)
    consensus += first_residue

for key in seqs.keys():
  print(key[0] + '\t' + key[1] + '\t' + ','.join(map(lambda x: str(x), out_indices[key])))

print('Consensus: ' + consensus)
