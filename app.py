import tensorflow_hub as hub
import numpy as np
import time
import logging
import os
from annoy import AnnoyIndex
import json



model = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2")
print("Model loaded successfully.")

# annoy settings
f = 512  # Dimension of embedding
index = AnnoyIndex(f, 'angular')


def get_embedding(text):
	embeddings = model([text])
	return np.array(embeddings[0])


def main ():
	with open('data.json' , 'r') as f:
		video_data = json.load(f)
	# for count, video in enumerate(video_data):
	# 	embedding  = get_embedding(video['title'])
	# 	nearest_neighbors = index.get_nns_by_vector(embedding, 3)
	# 	print(count )
	# 	index.add_item(count, embedding)

	# index.build(10)  # Adjust number of trees as needed
	# index.save('video_embeddings.ann')
	index.load('video_embeddings.ann')
	print("Annoy index updated and saved successfully.")


	# with open('video_links.json' , 'r') as f:
	# 	test_data = json.load(f)
	# start_load = time.time()

	# for count, video in enumerate(test_data):
	# 	embedding  = get_embedding(video['title'])
	# 	nearest_neighbors = index.get_nns_by_vector(embedding, 3)
	# 	print(video_data[nearest_neighbors[0]]['title'] , video_data[nearest_neighbors[0]]['category'] , count)
	# 	# print(nearest_neighbors[0])
	# end_load = time.time()
	# print(f"Time taken to load video data: {end_load - start_load:.4f} seconds")

	with open('video_links.json', 'r') as f:
			test_data = json.load(f)

	# Batch processing for nearest neighbors
	batch_size = 10 # Adjust batch size as needed
	start_processing = time.time()
	for i in range(0, 200
	, batch_size):
			batch_embeddings = []
			for video in test_data[i:i + batch_size]:
					embedding = get_embedding(video['title'])
					batch_embeddings.append(embedding)
			
			for j, embedding in enumerate(batch_embeddings):
					nearest_neighbors = index.get_nns_by_vector(embedding, 3)
					print(video_data[nearest_neighbors[0]]['title'], video_data[nearest_neighbors[0]]['category'], i + j)

	end_processing = time.time()
	print(f"Time taken for processing: {end_processing - start_processing:.4f} seconds")


	# given title
	# title = "Playing chess"

	# # get embedding
	# embedding = get_embedding(title)
	# nearest_neighbors = index.get_nns_by_vector(embedding, 3)
	# print(nearest_neighbors)
	# print(video_data[nearest_neighbors[0]]['title'] , video_data[nearest_neighbors[0]]['category'])





main()