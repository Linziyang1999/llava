import json
import numpy as np
import matplotlib.pyplot as plt

def load_attention_matrix(file_path):
    with open(file_path, 'r') as f:
        attention_list = json.load(f)
    return np.array(attention_list)

def plot_attention_mean(attention_matrix1,attention_matrix2, image_tokens=1876, output_file='attention_mean_compare.png'):
    # Extract the relevant part of the attention matrix
    # We are focusing on the attention provided by image tokens (rows) to non-image tokens (columns)
    attention_score = attention_matrix1[image_tokens+5:, :image_tokens]
    attention_score = attention_score.T
    # Compute the mean of attention scores provided by each image token
    mean_attention_score = np.mean(attention_score, axis=1)

    attention_score2 = attention_matrix2[image_tokens+5:, :image_tokens]
    attention_score2 = attention_score2.T
    # Compute the mean of attention scores provided by each image token
    mean_attention_score2 = np.mean(attention_score2, axis=1)
    # Plot the bar chart
    mean_score2 = np.mean(attention_score2)
    mean_score = np.mean(attention_score)
    print(mean_score)
    print(mean_score2)
    image_token_indices = range(image_tokens)
    plt.figure(figsize=(12, 6))
    
    plt.bar(image_token_indices, mean_attention_score)
    plt.axhline(mean_score, color='orange', linestyle='-', label='Mean Attention Score')
    #plt.bar(image_token_indices, mean_attention_score,color = 'darkgray')
    
    plt.ylim(0.0, 0.012) 
    plt.xlabel('Image Token Index')
    plt.ylabel('Mean Attention Score to Answer Tokens')
    plt.title('Mean Attention Scores Provided by Each Image Token to Answer Tokens')

    # Saving the plot to a file
    plt.savefig(output_file)
    plt.close()

# Load the attention matrix from the file
attention_matrix = load_attention_matrix('heatmap.json')
attention_matrix1 = load_attention_matrix('heatmap_vga.json')
# Plot the mean attention scores bar chart
plot_attention_mean(attention_matrix,attention_matrix1)