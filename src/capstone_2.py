# the basics
import pandas as pd
import numpy as np

# sklearn and scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import scipy.sparse as sp 

# other random libraries needed
from collections import Counter

# nltk and re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import string

# plotting functions
from plotting_functions import plot_lines_spoken

# Sentiment Analysis
#from nltk.sentiment.vader import SentimentIntensityAnalyzer

'''
This is the main python script for capstone 2, looking at a database of Seinfeld scripts

'''

def load_data():
    df_info = pd.read_csv('/Users/mattdevor/galvanize/capstone_2/data/seinfeld-chronicles/episode_info.csv', index_col=0)
    df_scripts = pd.read_csv('/Users/mattdevor/galvanize/capstone_2/data/seinfeld-chronicles/scripts.csv', index_col=0)

    # Fix season 1 episode 0 issue
    df_scripts.EpisodeNo = np.where(df_scripts.SEID =='S01E00', 0.0, df_scripts.EpisodeNo)

    # drop NAs
    df_scripts = df_scripts.dropna()
    
    return df_info, df_scripts

def lines_per_character(df_scripts, ep_number=None, season_number=None):
    if ep_number == None and season_number == None:
        character_lines = df_scripts['Character'].value_counts()
    else:
        character_lines = df_scripts[(df_scripts['EpisodeNo']==ep_number) & 
                                    (df_scripts['Season']==season_number)]['Character'].value_counts()
    return character_lines

def check_for_actions(df_scripts, column):
    # How many Character/Dialogue lines contain a parenthesis?
    return df_scripts[df_scripts[column].str.contains('(',regex=False)].count()

def check_for_punctuation(df_scripts, column):
    string.punctuation
    punct = '''[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]'''
    
    # How many Character/Dialogue lines contain Punctuation?
    return df_scripts[df_scripts[column].str.contains(punct)]

def strip_formatting(string):
    string = string.lower()
    string = re.sub(r"([.!?,'/()])", r"", string)
    return string

def preprocess_lines(lines):
    REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")
    lines = [REPLACE_NO_SPACE.sub("", line.lower()) for line in lines]
    lines = [REPLACE_WITH_SPACE.sub(" ", line) for line in lines]

    return lines

def get_actions(line):
    regex = re.compile(".*?\((.*?)\)")
    actions = re.findall(regex, line)
    return actions

def remove_actions(line):
    line = re.sub("[\(\[].*?[\)\]]", "", line)
    return line

def get_df_char_SEID(df_scripts, character, SEID):
    df_char_SEID = df_scripts[(df_scripts['Character']==character) & \
                              (df_scripts['SEID']==SEID)].copy().reset_index(drop=True)
    
    return df_char_SEID

def get_corpus_char_SEID(df_scripts, character, SEID):
    df_char_SEID = df_scripts[(df_scripts['Character']==character) & \
                              (df_scripts['SEID']==SEID)].copy().reset_index(drop=True)
    
    return df_char_SEID['Dialogue'].to_list()


def agg_dialogue_by_episode(df_scripts, df_info):
    df_scripts = df_scripts.copy()
    df_info = df_info.copy()

    all_documents = []
    df_cols = ['Dialogue', 'Lines_of_Dialogue', 'SEID', 'Season', 'Episode']
    df_new = pd.DataFrame(columns = df_cols)
    
    index = 0
    for SEID in df_scripts['SEID'].unique():
        dialogue = ' '.join(df_scripts[df_scripts['SEID']==SEID]['Dialogue'].to_list())
        lines_of_dialogue = int(df_scripts[df_scripts['SEID']==SEID]['EpisodeNo'].count())
        season = df_scripts[df_scripts['SEID']==SEID]['Season'].unique()[0]
        episode = df_scripts[df_scripts['SEID']==SEID]['EpisodeNo'].unique()[0]
        
        df_new.loc[index] = [dialogue, lines_of_dialogue, SEID, season, episode]
        
        index += 1

    merged = pd.merge(df_new, df_info.iloc[:,2:], on=['SEID']).reset_index(drop=True)

    merged['Lines_of_Dialogue'] = merged['Lines_of_Dialogue'].astype(int)

    return merged

def create_corpus_of_espisodes(df_docs_by_ep):
    df_docs_by_ep = df_docs_by_ep.copy()
    return df_docs_by_ep.Dialogue.values

if __name__=="__main__":
    df_info, df_scripts = load_data()
    main_characters = ['JERRY', 'GEORGE', 'ELAINE', 'KRAMER', 'NEWMAN']

    char_lines = lines_per_character(df_scripts)
    print(char_lines)
    char_lines = lines_per_character(df_scripts, 0.0, 1.0)
    print(char_lines)
    print(check_for_actions(df_scripts,'Dialogue'))
    print(check_for_punctuation(df_scripts, 'Dialogue'))

    # get character specific dfs and corpuses, based on Character and SEID
    df_jerry_s01_ep00 = get_df_char_SEID(df_scripts, main_characters[0], 'S01E00')
    print(df_jerry_s01_ep00)
    corpus_jerry_s01_ep00 = get_corpus_char_SEID(df_scripts, main_characters[0], 'S01E00')
    print(corpus_jerry_s01_ep00)

    # remove actions from corpus
    jerry_s01_ep00_corpus_a_r = [remove_actions(x) for x in corpus_jerry_s01_ep00]
    print(jerry_s01_ep00_corpus_a_r)

    # get all actions from this text
    jerry_s01_ep01_corpus_actions = [get_actions(x) for x in corpus_jerry_s01_ep00 if len(get_actions(x)) > 0]
    # flatten this list
    jerry_s01_ep01_corpus_actions = [item for sublist in jerry_s01_ep01_corpus_actions for item in sublist]
    print(jerry_s01_ep01_corpus_actions)

    # Two ways to clean data (may not need to do this depend on how mode is set up...)
    jerry_s01_ep01_corpus_a_r_clean = [strip_formatting(x) for x in jerry_s01_ep00_corpus_a_r]
    jerry_s01_ep01_corpus_a_r_clean = preprocess_lines(jerry_s01_ep00_corpus_a_r)

    # Tokenize and get count Vectorizer usking sklearn
    stop_words = text.ENGLISH_STOP_WORDS
    count = CountVectorizer(stop_words=stop_words)  #, max_features=3000, max_df = 1,  min_df=1)
    tf = count.fit_transform(corpus_jerry_s01_ep00)
    tf = sp.csr_matrix.toarray(tf)
    print(len(count.get_feature_names()))
    tokens_CV = count.get_feature_names()
    print(tokens_CV)

    # Use NLTK to tokenize Jerry's Dialogue from Episode 0
    punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%", "..."]
    stop_words = text.ENGLISH_STOP_WORDS.union(punc)
    dialogue = word_tokenize(' '.join(corpus_jerry_s01_ep00))
    nopunc = [word.lower() for word in dialogue if word not in stop_words]
    nopunc = ' '.join(nopunc)
    tokens = [word for word in nopunc.split()]
    print(len(set(tokens)))
    print(sorted(set(tokens)))

    # Plots commented out for now
    # plot_lines_spoken(df_scripts)

    ### Get a corpus of combined dialogue for all characters, by episode
    df_docs_by_ep = agg_dialogue_by_episode(df_scripts, df_info)
    print(df_docs_by_ep.head())
    print(df_docs_by_ep.Dialogue[0])
    print(' '.join(df_docs_by_ep.Dialogue[0]))

    ### LDA __TO DO __
    # create count vectorizor matrix, for comparison
    # corpus = 

    # count = CountVectorizer(stop_words=stop_words, max_features=1000, max_df = 0.85,  min_df=2)
    # tf = count.fit_transform(corpus)
    # tf = sp.csr_matrix.toarray(tf)
    
    # num_topics = 10
    # lda = LatentDirichletAllocation(n_components=num_topics, max_iter=5, learning_method='online',random_state=32, n_jobs=-1)
    # lda.fit(tf)

    # # phi is topics as rows and features as columns, which is the same as lda.components
    # phi = lda.components_
    # print(lda.components_.shape)

    # #theses are the words in our bag of words
    # tf_feature_names = count.get_feature_names() 

    # num_top_words = 10
    # display_topics(lda, tf_feature_names, num_top_words)
    
    print(df_docs_by_ep.info())
    print(df_docs_by_ep.head())
