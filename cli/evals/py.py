from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
import numpy as np


a = openai.Embedding.create(
    model='text-similarity-curie-001',
    input='The cat is on the mat.',
)

b = openai.Embedding.create(
    model='text-similarity-curie-001',
    input='',
    
)
print(a)
"""lista_floats_a = [float(element) for element in list(a['data'][0]['embedding'])]
lista_floats_b = [float(element) for element in list(b)]"""
"""a_avg1 = a.mean()
a_avg2 = b.mean()"""

similarity_score = cosine_similarity(np.array(a['data'][0]['embedding']).reshape(1, -1) , np.array(b['data'][0]['embedding']).reshape(1, -1))
print(similarity_score[0, 0])