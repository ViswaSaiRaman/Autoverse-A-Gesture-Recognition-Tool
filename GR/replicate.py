import os

signs = ['Ample','Best','Come','Done','Little','Ok','Run','Slow','Stop','Walk']

for i in(1,10):
    path = 'Dataset/'+signs[i]+'_Train'
    files = os.listdir(path)
    
    for index,file in enumerate(files):
            os.rename(os.path.join(path,file),os.path.join(path,signs[i]+'_'+str(index)+'.png'))
        
    
        