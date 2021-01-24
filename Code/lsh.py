import random
import operator
from pandas import DataFrame
from MinHash import *
from make_shingles import *
import time

def Hamming_Similarity(sign_mat, docA, docB) :
	''' takes the signature matrix as input along with the documents who's similarity is to be calculated

		return the Hamming Similarity of the documents
		Hamming Similarity = 1 - Hamming Distance
		Hamming Distance = no. of values that are different in the 2 columns / total number of values in each column
	'''
	count = 0
	vA = sign_mat.iloc[:,docA]
	vB = sign_mat.iloc[:,docB]  
	mask = vA!=vB
	count = np.sum(mask)

	n = 100 - count
	n /= 100
	return n


def LSH_from_sig(sign_mat, N, bucket_size, band_size, doc_num, bucket) :
	''' input signature matrix and number of permutations
		N = number of bands or number of hash functions to be generated
		bucket_size stores the number of buckets for each band
		band_size stores the number of rows in each band
		doc_num is the total number of documents in the signature matrix
		bucket is an empty 


		return a dictionary that stores the buckets corresponding to each band
		each bucket will have a key(hash value) and a set of documents that are hashed to it
		
	'''
	if len(sign_mat)%N != 0 :
		band_size += 1

	for x in range(0,N) :
		temp_bucket = {}
		start = x*band_size
		end = (x+1)*band_size
		# for doc in range(0,doc_num) :
		sum_hash = np.sum(sign_mat.iloc[start:end, :], axis = 0)
		
		# the sum of all values in a band corresponding to a document is hashed to the buckets for the particular band
		# this uses the N hash functions which are being generated randomly

		for doc, temp_val in enumerate(sum_hash):
			if temp_val in temp_bucket :
				temp_bucket[temp_val].add(doc)
			else :
				temp_bucket[temp_val] = set()
				temp_bucket[temp_val].add(doc)

		'''
		bucket[x] is a dictionary which has all the buckets for the band x

		each bucket has a key value which is the hashed result and a set as value, 
		this set contains the documents which are hashed to this bucket for the particular band
		''' 
		bucket[x] = temp_bucket

	return bucket

def ret_similar(sign_mat,threshold, doc_id) :
	'''
	takes signature matrix, threshold value for similarity and the document id for which similar documents are to be retrieved

	N = number of permutations/bands
	
	we have taken the number of buckets for each band to be 1,00,000

	the bucket dictionary returned from LSH_from_sig is processed to get all candidate pair of documents for the given document_id

	From the list of all the candidate pairs we check for the Hamming Similarity between them and the query document

	Only those documents with Similarity more than the threshold are stored in a separate dictionary ranked_res

	ranked_res is a dicitonary that uses similarity score as key and corresponding to each key there is a list of documents

	All the documents corresponding to the top 2 similarity scores are returned by the fn
	'''
	bucket_size = 100000
	N = 20
	band_size = int(len(sign_mat)/N)
	doc_num = sign_mat.shape[1]
	bucket = {}
	bucket = LSH_from_sig(sign_mat, N, bucket_size, band_size, doc_num, bucket)
	sim_docs = []
	for band in bucket :
		for hashval,doc_list in bucket[band].items() :
			if doc_id in doc_list :
				for doc in doc_list :
					if doc != doc_id :
						if doc not in sim_docs :
							sim_docs.append(doc)
	#print(sim_docs)
	ranked_res = dict()
	for doc in sim_docs :
		temp = Hamming_Similarity(sign_mat,doc,doc_id)
		if temp > threshold :
			if temp in ranked_res :
				ranked_res[temp].append(doc)
			else :
				ranked_res[temp] = []
				ranked_res[temp].append(doc)    
	ranked_res = dict(sorted(ranked_res.items(), key=operator.itemgetter(0),reverse=True))
	ranked_res = dict(list(ranked_res.items())[0:5])
	print(ranked_res)
	return ranked_res

if __name__ == '__main__':
	
	query = input('Input Query (Length should be atleast 5) : ')
	shingle_doc = make_shingles(r'human_data.txt', 5, query)	
	
	print(shingle_doc)

	doc_id = shingle_doc.shape[1]-1
	print(f'Document id is {doc_id}')

	sign_mat = generate_sign(r'shingle_doc_file', 100)
	
	print(sign_mat)

	start_time = time.time()
	sim_docs = ret_similar(sign_mat,0.85,doc_id)
	end_time = time.time()
	print(f'It took {end_time-start_time} seconds to give similarity scores')