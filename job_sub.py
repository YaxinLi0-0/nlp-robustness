import os
import time



num = [3,5]

command2run = 'sbatch direct_job.sb '
#command3run = 'sbatch influence_job.sb '

for s in num:
    command1 = command2run + str(s)
    print('run command: ', command1)
    os.system(command1)

#for s in num:
#    command1 = command3run + str(s)
#    print('run command: ', command1)
#    os.system(command1)



