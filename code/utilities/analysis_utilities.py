import os
import re

import matplotlib.patches as mpatches
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


def visualize_model_embeddings(mode, embeds, token_to_int, opts):
    """
    Plots the lemma or GloVe / POS / BERTembeddings for a few select 'representative' words
    """

    if mode == "word":
        if opts["use_glove_embeddings"] == True:   
            fig_name = "GloVe Embeddings"
        elif opts["use_glove_embeddings"] == False:   
            fig_name = "Model Embeddings"
    
        # pick some words to visualise
        words = ['dog', 'horse', 'animal', 'einstein', 'hitler','roberto', 'france',
        'italy', 'spain', 'donald', 'trump', 'education', 'harvard', 'mit', 'york', 'boston',
        "deutsche", "times", "bank", 'bush', "the", 'sometimes']


    elif mode == "pos":
        if opts["use_pos_embeddings"] == True:
            fig_name = "Model POS Embeddings"
        # pick some POS tags to visualise
        words = ["WRB", "MD", "WP", "WRB", "PRP", "PRP$", "NN", "NNS", "NNP", "NNPS", "JJ", "JJR", "JJS", "VB", "VBD", "VBG", "VBN", "VBP", "RB", "RBR", "RBS", "DT"] #"Paris", "France", "Europe", "united_states_of_america", "country", "city"]

    elif mode == "bert":
        assert opts["use_bert"] == True
        fig_name = "BERT Contextualized Embeddings"
        words = ['[UNK]', 'electric', '[UNK]', 'say', 'it', 'plan', 'to', 'increase', 'production', 'of', 'computer', 'memory', 'devices', 'on', 'a', 'large', 'scale', 'in', 'the', '[UNK]', 'and', 'japan', '.', 
        '[UNK]', 'electric', '[UNK]', 'say', 'it', 'plan', 'to', 'increase', 'production', 'of', 'computer', 'memory', 'devices', 'on', 'a', 'large', 'scale', 'in', 'the', '[UNK]', 'and', 'japan', '.', 
        '[UNK]', 'electric', '[UNK]', 'say', 'it', 'plan', 'to', 'increase', 'production', 'of', 'computer', 'memory', 'devices', 'on', 'a', 'large', 'scale', 'in', 'the', '[UNK]', 'and', 'japan', '.', 
        '[UNK]', 'electric', '[UNK]', 'say', 'it', 'plan', 'to', 'increase', 'production', 'of', 'computer', 'memory', 'devices', 'on', 'a', 'large', 'scale', 'in', 'the', '[UNK]', 'and', 'japan', '.']
    
    # perform PCA to reduce our 300d embeddings to 2d points that can be plotted
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(embeds.detach().cpu())

    indexes = [token_to_int[x] for x in words]    
    points = [pca_result[i] for i in range(len(indexes))]
    colors = ["blue"] * 23

    # If we are using BERT, prepare different colors and legend labels for points
    if mode == "bert":
        colors.extend(["red"] * 23)
        colors.extend(["yellow"] * 23)
        colors.extend(["green"] * 23)
        
        legend_label = ["None"] * 23
        legend_label.extend(["say"] * 23)
        legend_label.extend(["plan"] * 23)
        legend_label.extend(["increase"] * 23)


    for i,(x,y) in enumerate(points):
        plt.scatter(x, y, color=colors[i]) 
        plt.text(x, y, words[i], fontsize=12) # add a point label, shifted wrt to the point
    plt.title(fig_name)
    if mode == "bert":
        b_patch = mpatches.Patch(color='blue', label='None')
        r_patch = mpatches.Patch(color='red', label='say')
        y_patch = mpatches.Patch(color='yellow', label='plan')
        g_patch = mpatches.Patch(color='green', label='increase')
        plt.legend(handles=[b_patch, r_patch, y_patch, g_patch])
    plt.savefig("{}/{}.png".format(opts["save_model_path"], fig_name))
    plt.show()

    return
