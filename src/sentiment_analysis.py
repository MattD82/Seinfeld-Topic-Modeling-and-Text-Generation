import re
import numpy as np
import pandas as pd
from pprint import pprint

from tabulate import tabulate

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import capstone_2 as cap
import plotting_functions as pf 

global main_characters
main_characters = ['JERRY', 'GEORGE', 'ELAINE', 'KRAMER']

def sentiment_analyzer_scores(sentence):
    # basic sentiment analyzer for a single sentence
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(sentence)

    print("{:-<40} {}".format(sentence, str(score)))


def sentiment_analyzer(df_scripts):
    df = df_scripts.copy().reset_index(drop=True)
    
    sentiments = []
    
    # Sentiment Analysis using VADER
    analyzer = SentimentIntensityAnalyzer()

    for i in range(df.shape[0]):
        # feed in one line at a time to the sentiment analyzer
        line = df['Dialogue'].iloc[i]
        sentiment = analyzer.polarity_scores(line)
        
        # append to list of sentiments
        sentiments.append([sentiment['neg'], sentiment['pos'],
                           sentiment['neu'], sentiment['compound']])
        
    # create new columns for sentiments
    df[['neg', 'pos', 'neu', 'compound']] = pd.DataFrame(sentiments)
    
    # don't want 0's to be counted as positive or negative, since many neutral statements
    df['Negative'] = df['compound'] < -0.1
    df['Positive'] = df['compound'] > 0.1

    return df


def sentiment_by_episode(df_sentiment, df_info):
    df = df_sentiment.copy()

    df_lines_per_season = df.groupby(['SEID'], as_index=False)['Dialogue'].count()
    
    df_sum_negative_lines = df.groupby(['SEID'],as_index=False)['Negative'].sum()
    df_sum_positive_lines = df.groupby(['SEID'],as_index=False)['Positive'].sum() 


    df_merged = df_lines_per_season.merge(df_sum_negative_lines)
    df_merged = df_merged.merge(df_sum_positive_lines)
    
    df_merged['Percent_Negative'] = df_merged['Negative'] / df_merged['Dialogue']
    df_merged['Percent_Positive'] = df_merged['Positive'] / df_merged['Dialogue']
    
    df_merged = df_merged.merge(df_info)
    
    return df_merged


def season_sentiment(df_sentiment):
    # calculate statistics for sentiment - Entire Season
    positives_series = df_sentiment['Positive'].sum()
    negatives_series = df_sentiment['Negative'].sum()
    total_series = df_sentiment.shape[0]
    print(f"Positive lines for entire series: {positives_series / total_series * 100:.0f}%")
    print(f"Negative lines for entire series: {negatives_series / total_series * 100:.0f}%")
    print(f"Neutral lines for entire series: {(total_series - positives_series - negatives_series) / total_series * 100:.0f}%")
    

def most_pos_neg(df_sentiment_by_episode, num_eps):
    cols_for_table = ['Title', 'AirDate', 'Writers', 'Director', 'Percent_Positive', 'Percent_Negative']
    
    # Most Negative Episodes of Seinfeld
    most_negative = df_sentiment_by_episode.nlargest(num_eps, 'Percent_Negative')[cols_for_table]
    print("Episodes with Highest Percentage of Negative lines")
    print(tabulate(most_negative, tablefmt="pipe", headers="keys"))

    # Least Negative Episodes of Seinfeld
    least_negative = df_sentiment_by_episode.nsmallest(num_eps, 'Percent_Negative')[cols_for_table]
    print("Episodes with Lowest Percentage of Negative lines")
    print(tabulate(least_negative, tablefmt="pipe", headers="keys"))

    # Most Positive Episodes of Seinfeld
    most_positive = df_sentiment_by_episode.nlargest(num_eps, 'Percent_Positive')[cols_for_table]
    print("Episodes with Highest Percentage of Positive lines")
    print(tabulate(most_positive, tablefmt="pipe", headers="keys"))

    # Least Positive Episodes of Seinfeld
    least_positive = df_sentiment_by_episode.nsmallest(num_eps, 'Percent_Positive')[cols_for_table]
    print("Episodes with Lowest Percentage of Positive lines")
    print(tabulate(least_positive, tablefmt="pipe", headers="keys"))


def sentiment_by_season(df_sentiment_by_episode):
    df = df_sentiment_by_episode.copy()

    df_lines_per_season = df.groupby(['Season'], as_index=False)['Dialogue'].sum()
    
    df_sum_negative_lines = df.groupby(['Season'],as_index=False)['Negative'].sum()
    df_sum_positive_lines = df.groupby(['Season'],as_index=False)['Positive'].sum() 


    df_merged = df_lines_per_season.merge(df_sum_negative_lines)
    df_merged = df_merged.merge(df_sum_positive_lines)
    
    print(df_merged.head(5))
    
    df_merged['Percent_Negative'] = df_merged['Negative'] / df_merged['Dialogue']
    df_merged['Percent_Positive'] = df_merged['Positive'] / df_merged['Dialogue']
    
    #df_merged = df_merged.merge(df_info)
    
    return df_merged


def plot_sentiment_by_season(df_sentiment_by_season):
    df = df_sentiment_by_season.copy()
    
    fig = plt.figure(figsize=(12,6))
    
    x = df['Season']
    y_pos = df['Percent_Positive']
    y_neg = df['Percent_Negative']
        
    plt.scatter(x, y_pos, label='_nolegend_', color='g')
    plt.plot(x, y_pos, label='Positive Lines', lw=2.5, color='g')
    
    plt.scatter(x, y_neg, label='_nolegend_', color='r')
    plt.plot(x, y_neg, label='Negative Lines', lw=2.5, color='r', linestyle='--')

    plt.grid(linestyle="--", color=(0.75, 0.75, 0.75, 0.1))
    plt.ylabel('Percent Lines with Either Positive of Negative Sentiment')
    plt.xlabel('Season')
    plt.legend()
    plt.title('Positive and Negative Sentiment - by Season (%)', loc='center')
    plt.show()


def sentiment_by_character(df_sentiment):
    df = df_sentiment.copy()

    df_lines_per_season = df.groupby(['Season', 'Character'], 
                                      as_index=False)['Dialogue'].count()
    
    df_sum_negative_lines = df.groupby(['Season', 'Character'],
                                        as_index=False)['Negative'].sum()
    df_sum_positive_lines = df.groupby(['Season', 'Character'],
                                        as_index=False)['Positive'].sum() 

    df_merged = df_lines_per_season.merge(df_sum_negative_lines)
    df_merged = df_merged.merge(df_sum_positive_lines)
    
    print(df_merged.head(5))
    
    df_merged['Percent_Negative'] = df_merged['Negative'] / df_merged['Dialogue']
    df_merged['Percent_Positive'] = df_merged['Positive'] / df_merged['Dialogue']
    
    return df_merged

def plot_character_sentiment(df_sentiment_by_character):
    df = df_sentiment_by_character.copy()
    
    fig = plt.figure(figsize=(12,6))
    
    for character in main_characters:

        character_df = df[df['Character'] == character]
        x = character_df['Season']
        y_pos = character_df['Percent_Positive']
        y_neg = character_df['Percent_Negative']
        
        plt.scatter(x, y_pos, label='_nolegend_', color=main_char_colors[character], lw=0.5)
        plt.plot(x, y_pos, label=character.title(), color=main_char_colors[character], lw=2.5)
        plt.plot(x, y_neg, label='_nolegend_', 
                           color=main_char_colors[character], 
                           lw=2.5,
                           linestyle="--")

    plt.ylabel('Percent Lines with Either Positive or Negative Sentiment')
    plt.xlabel('Season')
    plt.legend(loc='upper left')
    plt.title('Main Character Positive and Negative Sentiment - by Season - (dashed lines represet negative sentiment)'
              , loc='center')
    plt.grid(linestyle="--", color=(0.75, 0.75, 0.75, 0.1))
    plt.ylim(0.1, 0.45)
    plt.show()

if __name__ == "__main__":
    # load in data 
    df_info, df_scripts = cap.load_data()
    df_docs_by_ep = cap.agg_dialogue_by_episode(df_scripts, df_info)

    # calculate sentiment by episode and by season
    df_sentiment = sentiment_analyzer(df_scripts) 
    df_sentiment_by_episode = sentiment_by_episode(df_sentiment, df_info)
    df_sentiment_by_season = sentiment_by_season(df_sentiment_by_episode)

    # print stats for season and tables for most positive and negative episodes
    season_sentiment(df_sentiment)
    most_pos_neg(df_sentiment_by_episode, 5)

    # plot sentiment by season
    plot_sentiment_by_season(df_sentiment_by_season)

    # create color maps for each main character list
    main_char_colors = pf.create_colormap('Set2', main_characters)

    # plot sentiment by character
    df_sentiment_by_character = sentiment_by_character(df_sentiment)
    plot_character_sentiment(df_sentiment_by_character)
