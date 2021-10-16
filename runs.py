import os
import sys

problemsLocation = sys.argv[1]

poolOfProblems = os.listdir(problemsLocation)
print('Working on {} problems...'.format(len(poolOfProblems)), '\n')

for run in poolOfProblems:
   print('Handling problem:{}'.format(run), '\n')
   os.system('python gpu.py '+ problemsLocation+'/' + run + ' 1000 0 20 60 30')
