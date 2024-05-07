import os
import sys
import time

i = 0
while True:
  print('Starting round ' + str(i))
  time.sleep(5)
  os.system('python3 train_ochre.py ' + ' '.join(sys.argv[1:]))
  i += 1
