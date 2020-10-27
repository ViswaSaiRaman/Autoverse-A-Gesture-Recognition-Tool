from PIL import Image

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

for X in range(65,90):
    if X==74 or X==90:
        continue
    for i in range(0, 1001):
    
        resizeImage('Dataset/'+chr(X)+'_Train/'+chr(X)+'_'+ str(i) + '.png')


