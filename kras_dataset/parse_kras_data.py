import csv

with open('can-22-0804_supplementary_datasets_s1-s4_suppsds1-sds4.csv', 'r') as csvfile:
  with open('kras_data.csv', 'w') as out_file:
    csvfile.readline()
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader: 
      protein = row[5]
      if protein != 'KRAS':
        continue
      pdb = row[0]
      nuc_state = row[6]
      inhibitor = row[17]
      bound = row[8]
      if inhibitor != 'None' or bound != 'None':
        continue
      if nuc_state == '3P':
        is_active = True
      elif nuc_state == '2P':
        is_active = False
      else:
        continue
#      y32_pos = row[3]
#      y71_pos = row[4]
#      if y32_pos == 'Y32in' and y71_pos == 'Y71in':
#        is_active = True
#      elif y32_pos == 'Y32out' and y71_pos == 'Disordered':
#        is_active = False
#      else:
#        continue
      chain = pdb[-1]
      pdb = pdb[:-1].upper()
      print(pdb + '\t' + chain + '\t' + str(int(is_active)))
