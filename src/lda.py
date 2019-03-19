# the basics
import pandas as pd
import numpy as np

# sklearn and scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import LatentDirichletAllocation
import scipy.sparse as sp 

# import from NLTK
import nltk
from nltk.corpus import stopwords

import re

# import wordcloud
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

import capstone_2 as cap 


class SklearnTopicModeler(object):
    '''
    Uses sklearn LDA to model topics within the Seinfeld script corpus.
    This should be easily generalizable to any corpus
    Right uses tf matrices created using CountVectorizer 
    '''
    def __init__(self, corpus):
        self.corpus = corpus

    def clean_vectorize(self):
        # remove single quotes and convert to lower case
        self.corpus = [re.sub("\'", "", sent) for sent in self.corpus]
        self.corpus = [sent.lower() for sent in self.corpus]

def display_topics(model, feature_names, num_top_words):
    topic_dict = {}
    idx_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-num_top_words - 1:-1]]))
        
        topic_dict[topic_idx] = [feature_names[i] for i in topic.argsort()[:-num_top_words - 1:-1]]
        idx_dict[topic_idx] = topic.argsort()[:-num_top_words - 1:-1]
        
    return topic_dict, idx_dict

def display_articles(index, theta, titles, num_articles):
    article_chosen = titles[index]
    theta_dif = theta[index] 
    diff_mat = np.abs(np.sum(theta - theta_dif, axis=1))
    print("Article Chosen is: {}".format(article_chosen))
    print("\n".join([titles[i]
                        for i in diff_mat.argsort()[:num_articles+1]]))
          
        
def display_articles_cosine(index, theta, titles, num_articles):
    article_chosen = titles[index]
    theta_dif = theta[index] 
    diff_mat = pairwise_distances(theta_dif.reshape(1,-1),theta,metric='cosine')
    diff_mat = diff_mat.reshape(-1)
    print(diff_mat.argsort()[:num_articles+1])
    print("Title of Episode is: {}".format(article_chosen))
    print("\n".join([titles[i]
                        for i in diff_mat.argsort()[:num_articles+1]]))
    
def top_closest_episodes(episode, theta, titles, num_episodes):
    return titles[titles==episode].index[0]
    
    
def display_episodes_by_topics(theta, topic, titles, num_episodes):
    print("Topic Chosen is: {}".format(topic))
    print("\n".join(titles[theta[:, topic].argsort()[::-1][:num_episodes]]))
    


df_info, df_scripts = cap.load_data()
df_docs_by_ep = cap.agg_dialogue_by_episode(df_scripts, df_info)
corpus = df_docs_by_ep.Dialogue.values



# add to stop words
#more_stop_words = ['ya', 'ha', 'mr', 'okay', 'ah'] #<--- REMOVING 'alright' really changed things!
more_stop_words = ['ya', 'ha', 'mr', 'okay', 'ah', 'alright', 'apartment', 'talk', 
                   'happened', 'car', 'phone', 'looks', 'woman', 'getting', 'new', 
                   'day', 'talking', 'wanna', 'bad', 'love', 'looking', 'night',
                   'work', 'em', 'cmon', 'kind', 'god', 'coffee', 'friend', 'away', 'making']
stop_words = text.ENGLISH_STOP_WORDS.union(more_stop_words)

# create tf matrix from corpus - note this removes puncuation automatically
vectorizer = CountVectorizer(stop_words=stop_words, max_features=3000, max_df = 0.85,  min_df=2)
tf = vectorizer.fit_transform(corpus)
tf = sp.csr_matrix.toarray(tf)

# num_topics = 10 produced really interesting results
# create LDA model using sklearn
num_topics = 10
lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online',random_state=32, n_jobs=-1)
lda.fit(tf)

# phi is topics as rows and features (our tf-matrix in this case) as columns, which is the same as lda.components
phi = lda.components_

# theta relates total episodes (as rows) to topics (as columns)
theta = lda.transform(tf)

# theses are the words in our bag of words
tf_feature_names = vectorizer.get_feature_names() 

# get titles from df
titles = df_docs_by_ep.Title.values

lda.perplexity(tf)

# Choose number of words to display in each cloud
num_top_words = 10

ten_topics, idxs = display_topics(lda, tf_feature_names, num_top_words)

tot_dict = {}
new_dict = {}
for i, j in idxs.items():
    t_dict = {}
    for v in j:
        word = tf_feature_names[v]
        t_dict[word] = np.sum(tf[:, v])
        new_dict[i] = t_dict
        tot_dict.update(t_dict)

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
cloud = WordCloud(stopwords=stop_words,
                  background_color='black',
                  width=2500,
                  height=1800,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

#topics = lda.show_topics(formatted=False)

fig, axes = plt.subplots(1, 3, figsize=(10,5), sharex=True, sharey=True)
gs1 = gridspec.GridSpec(4, 4)
gs1.update(wspace=0.0, hspace=0.0)

for i, ax in enumerate(axes.flatten()[:3]):
    fig.add_subplot(ax)
    topic_words = new_dict[i]
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    # plt.gca().imshow(cloud)
    # plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
    # plt.gca().axis('off')

    plt.imshow(cloud)
    plt.title('Topic ' + str(i), fontdict=dict(size=16))
    plt.axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout(pad=0)

plt.show()