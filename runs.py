import os
import sys

problemsLocation = sys.argv[1]

poolOfProblems = os.listdir(problemsLocation)
print('Working on {} problems'.format(len(poolOfProblems)))
for run in poolOfProblems:
   os.system('python gpu.py ' + run + ' 20000 0 20 60 30')
