# Original author's code


from keras.models import model_from_json
import json

def create_model(path):
	with open(path) as infile:
		return json.load(infile)

def load_model_weights(model_arch,path):
	model = model_from_json(model_arch)
	model.load_weights(path)
	return model