import pickle
from pandas import DataFrame

def make_shingles(path_db, length, query = None):
	'''
	Creates shingles given the database
	Inputs : Path of documents, length of shingle, query string
	Returns : Shingle to document matrix
	'''
	shingles = {}
	total_docs = 0
	#Calculating total number of documents
	with open(path_db) as file:
		file.readline()
		for i, line in enumerate(file.readlines()):
			total_docs+=1

	print(f'Total number of documents = {total_docs}')
	#making shingle doument database
	with open(path_db) as file:
		file.readline()
		for i, line in enumerate(file.readlines()):
			data, _ = line.split()
			data = data.strip('N')
			if len(data)<length:
				continue
			for start in range(0, len(data)-length+1):
				subseq  = data[start:start+length]
				if subseq in shingles:
					shingles[subseq][i] = 1
				else: #new shingle
					shingles[subseq] = [0]*(total_docs+1 if query is not None else total_docs)
					shingles[subseq][i] = 1
	
	#If query string is given				
	if query is not None and len(query)>=length:
		for start in range(0, len(query)-length+1):
			subseq  = query[start:start+length]
			if subseq in shingles:
				shingles[subseq][total_docs] = 1
			else: #new shingle
				shingles[subseq] = [0]*(total_docs+1 if query is not None else total_docs)
				shingles[subseq][total_docs] = 1

	shingles = DataFrame(shingles).transpose()
	shingle_doc_file = open('shingle_doc_file', 'wb')
	pickle.dump(shingles, shingle_doc_file)
	shingle_doc_file.close()

	return shingles

if __name__ == '__main__':
	data = make_shingles(r'human_data.txt',5)
	print(data)