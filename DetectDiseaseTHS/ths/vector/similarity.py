import numpy as np

def cosine_similarity(u, v):

    dot = np.dot(u, v)
    # Compute the L2 norm of u and v
    norm_u = np.sqrt(np.dot(u, u))
    norm_v = np.sqrt(np.dot(v, v))
    # Compute the cosine similarity defined by formula (1) (≈1 line)
    result = dot / (norm_u * norm_v)

    return result
