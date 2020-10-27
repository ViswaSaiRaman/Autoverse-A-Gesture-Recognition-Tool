import os
path = 'Dataset/W_Train'
files = os.listdir(path)
i = 0

for file in files:
    os.rename(os.path.join(path, file), os.path.join(path, 'W_'+str(i)+'.png'))
    i = i+1

