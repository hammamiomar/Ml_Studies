import numpy as np
import matplotlib.pyplot as plt

def attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None):
    assert Q.shape[1] == K.shape[1], "Q and K must have the same feature dimension"
    assert K.shape[0] == V.shape[0], "K and V must have the same number of rows"

    scores = np.dot(Q, K.T)
    print(f'scores before reg: {scores}')
    scores = scores / np.sqrt(Q.shape[1])
    # need reg or else the variances add up. logits pre softmax will be too extreme and gradients will dissapear.
    print(f'scores after reg: {scores}')
    
    if mask is not None:
        scores = scores + (mask * -1e9)

    #axis 1--> scores need to sum each row into 1 col. numpy is row by col. so need to sum dim 1.
    weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    output = np.dot(weights, V)
    return output, weights

def plot_attention(weights):
    plt.figure(figsize=(8,6))
    plt.imshow(weights,cmap="Blues")
    plt.colorbar(label="Attention Weight")
    plt.title("Attention weights")
    plt.xlabel("Key positions")
    plt.ylabel("Query Positions")
    
    for i in range(weights.shape[0]):
        for j in range(weights.shape[1]):
            plt.text(j,i, f'{weights[i,j]:.3f}',ha="center",va="center")
    plt.show()
        
def causal_mask(size):
    return np.triu(np.ones((size,size)),k=1)

if __name__ == "__main__":
    Q = np.array([[1,4,3],
                [7,2,1],
                [4,6,8]])
    K = np.array([[4,3,7],
                    [8,6,1]])
    V = np.array([[8,5,1],
                    [4,3,1]])
    output, weights = attention(Q, K, V)
    print("Basic attention weights:")
    print(weights)
    plot_attention(weights)
    
    # Self-attention with causal mask
    seq = np.array([[1,2,3],
                    [4,5,6], 
                    [7,8,9]], dtype=float)
    
    mask = causal_mask(3)
    print(f"\nCausal mask:\n{mask}")
    
    output, weights = attention(seq, seq, seq, mask=mask)
    print("Causal attention weights:")
    print(weights)
    plot_attention(weights)
    
    
    
    
    