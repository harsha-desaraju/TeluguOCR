
import numpy as np
from utils import read_image
from utils import split_lines, split_words, split_chars
from utils import load_model, predict, preprocess_image


# Load models and encoding
char_model = load_model("./Training/model_chars_tccnn-l.json","./Training/model_chars_tccnn-l.hdf5")
vg_model = load_model("./Training/model_v_g.json","./Training/model_v_g_weights.hdf5")


char = {}
with open("./Encoding/char.txt",'r') as f:
	lst = f.readlines()
	for i in range(len(lst)):
		if i%2 == 0:
			char[(i/2)+1] = lst[i+1]

vg = {}
with open("./Encoding/vattu_gunintam.txt",'r') as f:
	lst = f.readlines()
	for i in range(len(lst)):
		if i%2 == 0:
			vg[(i/2)] = lst[i+1]



fin = "./Images/Img1.jpg"
img = read_image(fin)
img = preprocess_image(img)

with open("output.html",'w') as fout:
	for line in split_lines(img):
		for word in split_words(line):
			for letter in split_chars(word):
				c,v = predict(letter,char_model,vg_model)
				if b==-1:
					fout.write(char[c][:-2])
				else:
					fout.write(char[c][:-2]+vg[v][:-2])
			fout.write("&nbsp")
		fout.write("<br/>")
