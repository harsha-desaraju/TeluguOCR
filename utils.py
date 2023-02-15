
import cv2
import json
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from deskew import determine_skew
from skimage.transform import rotate
from keras.models import load_model
from keras.models import model_from_json


# Reads an image and returns an image as a 2d numpy array.
def read_image(path):
    return np.asarray(Image.open(path).convert('L'))


# Binarizes an image.The input is a 2d numpy array.
def Binarize_image(img):
    # Apply Gaussian filtering 
    filtered_img = cv2.GaussianBlur(img,(5,5),0)
    # Apply thresholding.Otsu's and binary.
    ret,Thresh_img = cv2.threshold(filtered_img,0,255,cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    return Thresh_img


# Expects a binarized image as input
def skew_corrected_img(img):
    angle = determine_skew(img)
    return rotate(img,angle,resize=True,mode='edge')*255


def preprocess_image(img):
	img = Binarize_image(img)
	img = skew_corrected_img(img)
	return img


# Shows the images after different steps of preprocessing
def show_processed_images(img):

	fig, (ax1,ax2,ax3) = plt.subplots(1,3)
	ax1.imshow(img,cmap='Greys')
	ax1.set_title('Original Image')

	img = Binarize_image(img)
	ax2.imshow(img,cmap='Greys')
	ax2.set_title('Binarized Image')

	img = skew_corrected_img(img)
	ax3.imshow(img,cmap='Greys')
	ax3.set_title('Skew Corrected Image')

	plt.show()


# For splitting the image into lines.
def split_lines(img):

	def split_lines_d(img):
	    x=[]
	    start = 0
	    stop = 0
	    for i in range(len(img)-1):
	        if np.count_nonzero(img[i]==0) > 0 and np.count_nonzero(img[i-1]==0) == 0 :
	            start = i
	        #if i!=len(img)-1:
	        if np.count_nonzero(img[i]==0) > 0 and np.count_nonzero(img[i+1]==0) == 0 :
	            stop = i
	            x.append(stop-start)      
	    return x

	def min_dist(img):
		splited_dist = split_lines_d(img)
		sort_dist=sorted(splited_dist)
		diff=[]
		for i in range(len(sort_dist)-1):
			z=sort_dist[i+1]-sort_dist[i]
			diff.append([z,sort_dist[i]])
		diff2=sorted(diff)
		if diff2[-1][0]<sort_dist[0]:
			dist=sort_dist[0]-2
		else:
			dist=diff2[-1][1]
		return dist

	def split_lines_k(img):
	    x=[]
	    start = 0
	    stop = 0
	    for i in range(len(img)-1):
	        if np.count_nonzero(img[i]==0) > 0 and np.count_nonzero(img[i-1]==0) == 0 :
	            start = i
	        if np.count_nonzero(img[i]==0) > 0 and np.count_nonzero(img[i+1]==0) == 0 :
	            stop = i
	            x.append([stop,start])
	    return x

	def forming_lines(img):
		new=[]
		img2=split_lines_k(img)
		new.append([img2[0][1],img2[0][0]])
		for i in range(len(img2)-1):
			if img2[i+1][0]-img2[i+1][1]<=min_dist(img):
				if new[-1][0]==img2[i][1]:
					new[-1][1]=img2[i+1][0]
				else:
					new.append([img2[i][1],img2[i+1][0]])
			else:
				new.append([img2[i+1][1],img2[i+1][0]])
		return new

	line_images=[]
	points=forming_lines(img)
	for i in points:
		start=i[0]
		stop=i[1]
		line_images.append(img[start:stop+1])
	return line_images



# For splitting an image of a line into words
def split_words(img):
    # Outputs an array containing no. of zeroes in each row
    def num_zero(img):
        temp_list = []
        for i in range(len(img)):
            temp_list.append(np.count_nonzero(img[i]==0))
        return np.array(temp_list)
    
    # Image to be trimmed at the top and bottom. 
    # arr argument is the array containing no. of zeroes in each row
    def trim_image(img,arr):
        start = 0
        stop = 0
        for i in range(len(arr)):
            if arr[i]!=0:
                start = i
                break
        for i in reversed(range(len(arr))):
            if arr[i] != 0:
                stop = i
                break
        return (img[start:stop+1],arr[start:stop+1])

    
    # Returns length of consecutive zeroes
    def length(arr,ind,ele):
        if ind != len(arr)-1:
            i = ind
            while i<=len(arr)-1 and arr[i]==ele:
                i+=1
                
            if i != ind:
                return i-ind
            else:
                return 1
        else:
            return 

    # Returns True if no word exists after the current row.
    def EOL(arr,ind,ele):
        temp = True
        for i in range(ind,len(arr)):
            if arr[i] != 0:
                temp = False
        return temp

    # Returns the length above which the image should be split
    # The argument is the array of no. of zeroes
    def limiting_length(arr):
        temp_list = []
        i=0
        while i< len(arr):
            temp=0
            if arr[i] == 0:
                temp = length(arr,i,0)
                temp_list.append(temp)
            i=i+temp+1

        # maximum difference between consecutive elements
        temp_list.sort()
        diff = 0
        lim_len = 0
        for i in range(1,len(temp_list)):
            temp = temp_list[i] - temp_list[i-1]
            if temp > diff:
                diff = temp
                lim_len = temp_list[i-1]
        return lim_len
    
    img = np.transpose(img)
    zeroes = num_zero(img)
    trim_img, zeroes = trim_image(img,zeroes) 
    limit_len = limiting_length(zeroes)
    splited_images = []
    start = 0
    stop = 0
    i=1
    
    while i<len(zeroes):
        if zeroes[i] != 0:
            start = i
            while i < len(zeroes):
                if zeroes[i]!=0:
                    i+=1
                else:
                    if length(zeroes,i,0) > limit_len:
                        break
                    else:
                        i+=length(zeroes,i,0)
            if length(zeroes,i,0) > limit_len or EOL(zeroes,i,0):
                stop = i
                temp_img = np.transpose(trim_img[start:stop])
                splited_images.append(temp_img)
                i+=limit_len
        else:
            i+=1
    return splited_images


# For splitting an image of a word into characters
def split_chars(img1):
    img = np.transpose(img1)
    splitted_letters = []
    i=0
    while i < len(img):
        if np.count_nonzero(img[i]==0) != 0:
            start = i
            j=i+1
            while j < len(img):
                if np.count_nonzero(img[j]==0) == 0:
                    break
                j+=1
            temp_img = np.transpose(img[start:j])
            splitted_letters.append(temp_img)
            i=j
        else:
            i+=1
    return splitted_letters


# Loads model
def load_model(json_path,weights_path):
	with open(json_path,'r') as jf:
		json_file = json.load(jf)

	model = model_from_json(json_file)
	model.load_weights(weights_path)
	return model


# This function predicts the character from the image
def predict(char,model_1,model_2):
  img=char
  #img=cv2.transpose(img)
  img=cv2.resize(img,(32,32))
  img = np.asarray(img).reshape(1,1,32,32)
  img = img.astype('float32')
  img = img/255.0
  chars=[]
  vg=[]
  print(img.shape)
  out =  model_1.predict(img)
  chars.append(np.where(out==out.max())[1][:]+1)
  if np.where(out==out.max())[1][:]+1>=20 and np.where(out==out.max())[1][:]+1<=55:
    out2 = model_2.predict(img)
    vg.append(np.where(out2==out2.max())[1][:])
  else:
    vg.append(np.array([-1]))
  return chars[0][0],vg[0][0]





if __name__ == '__main__':
	img = read_image("./Images/Img2.png")
	# processed_images(img)

	img = preprocess_image(img)
	lines = split_lines(img)

	words = split_words(lines[0])

	chars = split_chars(words[0])

	for c in chars:
		plt.imshow(c,cmap='Greys')
	plt.show()




