import os
import cv2
from tqdm import tqdm


DATADIR =r'D:\Downloads\Dataset\New folder'#---CHANGE it to your SOurce directory

CATEGORIES = ["Healthy", "Unhealthy"]


IMG_SIZE = 100
i=1
for category in CATEGORIES:  

        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category)  
        if(class_num==0):
            writepath=r'D:\Downloads\Healthy' # ---------CHANGE it to your directory where you want to save the new images
        else:
            writepath=r'D:\Downloads\Unhealthy'#-----CHANGE it to your directory where you want to save the new images
            
        for img in tqdm(os.listdir(path)): 
            addname=r'\img'+str(i)+'.JPG'
            i+=1
            wpath=writepath+addname
                    
            img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)  
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            cv2.imwrite(wpath,new_array)
                
