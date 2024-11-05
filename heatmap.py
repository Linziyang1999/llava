import json
import numpy as np
import matplotlib.pyplot as plt

def load_attention_matrix(file_path):
    with open(file_path, 'r') as f:
        attention_list = json.load(f)
    return np.array(attention_list)

def plot_attention_contribution(attention_matrix, image_tokens=1876, output_file='attention_contribution_compare.png'):
    # Extract the relevant part of the attention matrix
    # We are focusing on the attention received by non-image tokens (rows) from image tokens (columns)
    attention_score = attention_matrix[image_tokens+5:, :image_tokens]

    # Compute the sum of attention scores received by each non-image token
    sum_attention_score = np.sum(attention_score, axis=1)

    mean_attention_score = np.mean(sum_attention_score)

   

    # Plot the bar chart
    non_image_token_indices = range(len(sum_attention_score))
    plt.figure(figsize=(12, 6))
    plt.bar(non_image_token_indices, sum_attention_score, color='darkgray')
    plt.axhline(mean_attention_score, color='orange', linestyle='-', label='Mean Attention Score')
    plt.xlabel('Answer Token Index')
    plt.ylabel('Sum of Attention Scores from Image Tokens')
    plt.title('Sum of Attention Scores Received by Each Answer Token from Image Tokens')

    # Saving the plot to a file
    plt.savefig(output_file)
    plt.close()

# Load the attention matrix from the file
attention_matrix = load_attention_matrix('heatmap.json')

# Plot the attention contribution bar chart
plot_attention_contribution(attention_matrix)