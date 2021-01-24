import pickle
import numpy as np
import random
import time
from tqdm import tqdm
from pandas import DataFrame

def first_non_zero(shingle_doc):
	'''
	Returns the first non zero element's index, operation is performed for each column
	Input : Shuffled shingle document
	'''
	mask = shingle_doc>0
	return np.where(mask.any(axis = 0), mask.argmax(axis = 0), -1)


def generate_sign(shingle_doc_path, num_of_perms): 
	'''
	Generates signature matrix 
	Inputs : Path of shingle document, number of hash required
	Returns : Signature Matrix
	'''
	start_time = time.time()
	
	file = open(shingle_doc_path,'rb')
	shingle_doc = pickle.load(file)
	file.close()

	shingle_doc = np.array(shingle_doc)

	num_of_shingles, num_of_docs = shingle_doc.shape 

	signature = np.zeros((1, num_of_docs)).astype(int)
	perm_list = set()
	for i in range(num_of_perms):
		np.random.shuffle(shingle_doc)
		temp = first_non_zero(shingle_doc).astype(int)
		signature = np.vstack((signature, temp))
	signature = np.delete(signature, 0, axis= 0)
	
	end_time = time.time()
	print(f'It took {end_time-start_time} seconds to generate signature matrix')
	signature = DataFrame(signature)
	return signature

if __name__ == '__main__':
	print(generate_sign(r'shingle_doc_file', 100))