import torch
import torch.nn as nn

#suppose we have 5 food items : ['sandwich', 'soup', 'salad', 'pasta', 'pizza' ]

vocab_size = 5 #number of unique food items
embedding_dim = 3 #each food will be represented by a 3-dim vector space

#creating the embedding layer
embedding = nn.Embedding(vocab_size, embedding_dim)

#let's say we want to embedding for 'soup' which is at index 1
food_index = torch.tensor([1]) #index of 'soup'
food_embedding = embedding(food_index)

print("Embedding for 'soup':", food_embedding)

foods = torch.tensor([0, 2, 4]) #indices for 'sandwich', 'salad', 'pizza'
food_embeddings = embedding(foods)

print("Embeddings for 'sandwich', 'salad', 'pizza':", food_embeddings)


'''
What’s Happening Here?

Instead of [0,1,0,0,0] (one-hot for hotdog),
PyTorch gives you a dense 3D vector like [0.25, -0.17, 0.88].

These numbers are learnable weights — updated when training your model.

Foods that appear in similar contexts (shawarma & hotdog) will get closer embeddings.

'''