import tensorflow_hub as hub
import numpy as np
import logging
import os
from annoy import AnnoyIndex
import json


# model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

model = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2")
print("Model loaded successfully.")



def get_embedding(text):
    embeddings = model([text])
    return np.array(embeddings[0])
f = 512  # Dimension of embedding
index = AnnoyIndex(f, 'angular')




def main():
    # Load video titles from JSON file
    with open('video_links.json', 'r') as f:
        video_data = json.load(f)

    with open('video_links.json', 'r') as f:
        index_to_category = json.load(f)
    # Create Annoy index


    # # Add titles to Annoy index
    # for video in video_data:
    #     embedding = get_embedding(video['title'])
    #     index.add_item(video['title'], embedding)
    for count, video in enumerate(video_data):
      print(f"Processing video {count + 1}/{len(video_data)}")  # Print the loop count
      embedding = get_embedding(video['title'])
      index.add_item(count, embedding)
      index.build(10)  # 10 trees for Annoy

    # Save the index
    index.save('video_embeddings.ann')
    title="How to build a machine learning model"
    embedding = get_embedding(title)
    
    
    def check_video_category(title, index, index_to_category):
        embedding = get_embedding(title)
        nearest_neighbors = index.get_nns_by_vector(embedding, 3)
    
        # Check if any nearest neighbor belongs to the tech category
        tech_category = 'tech'
        is_tech = any(index_to_category[str(neighbor)] == tech_category for neighbor in nearest_neighbors)
    
        return is_tech, nearest_neighbors
    # nearest_neighbors = index.get_nns_by_vector(embedding, 3)
    is_tech, nearest_neighbors = check_video_category(title, index, index_to_category)
    # tech_ids = [1, 2]  # IDs of tech videos
    # is_tech = any(neighbor in tech_ids for neighbor in nearest_neighbors)
    
    print({"is_tech": is_tech, "nearest_neighbors": nearest_neighbors})

    # Build the index
    # with open('video_data.json', 'r') as f:
    #     video_data = json.load(f)

    # # Process each video title
    # for video in video_data:
    #     title = video['title']
    #     is_tech, nearest_neighbors = check_video_category(title, index, index_to_category)

    #     print(f"The video '{title}' is {'tech' if is_tech else 'non-tech'}.")
    #     print({"is_tech": is_tech, "nearest_neighbors": nearest_neighbors})
    #     print()
    # print("Index built and saved successfully.")

main()
