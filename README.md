![](images/header.png)
# A Capstone about "Nothing"
Capstone Project for Galvanize Data Science Immersive 

by Matt Devor

Important metrics
Clarity of Project Description (0-3)    
Data description and EDA (0-3)    
Modeling Methodology and Validation (0-3)    
Results and Future Work (0-3)   
Wow! factor (0-3)

## Table of Contents



# Introduction
Seinfeld is an American television sitcom that ran for nine seasons on NBC, from 1989 to 1998. It was created by Larry David and Jerry Seinfeld, with the latter starring as a fictionalized version of himself. Set predominantly in an apartment building in Manhattan's Upper West Side in New York City, the show features a handful of Jerry's friends and acquaintances, including best friend George Costanza, friend and former girlfriend Elaine Benes, and neighbor across the hall Cosmo Kramer (Michael Richards). It is often described as being "a show about nothing", as many of its episodes are about the minutiae of daily life.

![](images/nothing.jpg)


I have been a huge fan of this show for around two decades, and still find myself going back and enjoying old episodes. I think the writing is absolutely brilliant, still holds up (even if some references are dated), and really appreciate the way that Larry David in particular can transform seemingly mundane daily occurrences into entertaining dialogue. 

I discovered a database containing scripts for every Seinfeld episode (located [here](https://www.kaggle.com/thec03u5/seinfeld-chronicles)), with dialogue separated by character. Since we have learned so much about NLP recently, and because NLP has so many practical applications, I thought this would be a fun corpus to take on, and try to extract meaning from. I also thought it would be a very interesting problem to try and see if I can train a model to emulate the speech patterns of the four main characters mentioned above, and even create a model that would “predict”/generate the script for a new episode of Seinfeld.




[Back to Top](#Table-of-Contents)

# Strategy and Process
- Overview of Data
- Exploratory Data Analysis
- Unsupervised Learning: LDA - Sklearn
- Unsupervised Learning: LDA - Gensim/Spacy
- Sentiment Analysis
- Text Generation
- Reflection and Future Work

# Overview of the Data
- This dataset consists of two databases:
  - The first is 174 rows: a row for each episode of Seinfeld created, and includes features such as season, episode #, Title, AirDate, Writer(s), and Director.  
  - The second is ~54,600 rows: a row for each line of dialogue spoken by each character within each episode, and includes features such as Character (who is speaking), Dialogue (what they say), EpisodeNo, SeasonEpisodeID, and Season. 
- There are 9 seasons of Seinfeld, with the pilot airing on June 7th, 1989, and the final episode airing May 14th, 1998.
- There are only 10 NaNs in the dataset, and there is no dialogue info for those rows, so I decided it was ok to drop them.
- There are an average of ~17,000 alphanumeric characters in each Seinfeld episode script, and a total of 2,920,645 characters in the entire series.
- Depending on how the text is cleaned tokenized, and stemmed/lemmatized, there are approximately 8,500 unique words in the corpus.


[Back to Top](#Table-of-Contents)

# Exploratory Data Analysis
![](images/lines_by_char_series.png)

![](images/lines_by_char_season.png)

![](images/writers_by_season.png)

![](images/writers_series.png)

![](images/avg_dialogue_chars.png)

## EDA Takeaways
- 


[Back to Top](#Table-of-Contents)

# Unsupervised Learning: LDA - Sklearn
![](images/lda_sklearn_wordcloud.png)



# Unsupervised Learning: Gensim/Spacy
![](images/lda_gensim_wordcloud.png)

![](images/lda_gif.gif)


# Sentiment Analysis

![](images/problem.gif)

Positive lines for entire series: 31%
Negative lines for entire series: 16%
Neutral lines for entire series: 53%

##  Most Negative Episodes of Seinfeld

|   Season - ADD  | Title             | AirDate   | Writers                     | Director      |   Percent_Positive |   Percent_Negative |
|----:|:------------------|:----------|:----------------------------|:--------------|-------------------:|-------------------:|
|   5 | The Ex-Girlfriend | 16-Jan-91 | Larry David, Jerry Seinfeld | Tom Cherones  |           0.318367 |           0.24898  |
|   6 | The Pony Remark   | 30-Jan-91 | Larry David, Jerry Seinfeld | Tom Cherones  |           0.28     |           0.24     |
| 134 | The Little Kicks  | 10-Oct-96 | Spike Feresten              | Andy Ackerman |           0.287037 |           0.234568 |
|  14 | The Baby Shower   | 16-May-91 | Larry Charles               | Tom Cherones  |           0.244635 |           0.23176  |
|   7 | The Jacket        | 6-Feb-91  | Larry David, Jerry Seinfeld | Tom Cherones  |           0.287356 |           0.229885 |

## Least Negative Episodes of Seinfeld

|     | Title                                     | AirDate   | Writers                                    | Director           |   Percent_Positive |   Percent_Negative |
|----:|:------------------------------------------|:----------|:-------------------------------------------|:-------------------|-------------------:|-------------------:|
|  93 | The Secretary                             | 8-Dec-94  | Carol Leifer, Marjorie Gross               | David Owen Trainor |           0.406154 |          0.0984615 |
|  77 | The Marine Biologist                      | 10-Feb-94 | Ron Hague, Charlie Rubin                   | Tom Cherones       |           0.396011 |          0.0997151 |
| 164 | The Reverse Peephole (a.k.a. The Man Fur) | 15-Jan-98 | Spike Feresten                             | Andy Ackerman      |           0.328313 |          0.10241   |
|  73 | The Cigar Store Indian                    | 9-Dec-93  | Tom Gammill, Max Pross                     | Tom Cherones       |           0.330383 |          0.106195  |
|  84 | The Opposite                              | 19-May-94 | Andy Cowan and Larry David, Jerry Seinfeld | Tom Cherones       |           0.351171 |          0.107023  |

## Most Positive Episodes of Seinfeld

|    | Title               | AirDate   | Writers                      | Director           |   Percent_Positive |   Percent_Negative |
|---:|:--------------------|:----------|:-----------------------------|:-------------------|-------------------:|-------------------:|
| 40 | The Trip (1)        | 12-Aug-92 | Larry Charles                | Tom Cherones       |           0.422145 |          0.124567  |
|  0 | Good News, Bad News | 5-Jul-89  | Larry David, Jerry Seinfeld  | Art Wolff          |           0.417062 |          0.146919  |
| 82 | The Fire            | 5-May-94  | Larry Charles                | Tom Cherones       |           0.414966 |          0.14966   |
| 54 | The Visa            | 27-Jan-93 | Peter Mehlman                | Tom Cherones       |           0.408027 |          0.167224  |
| 93 | The Secretary       | 8-Dec-94  | Carol Leifer, Marjorie Gross | David Owen Trainor |           0.406154 |          0.0984615 |

## Least Positive Episodes of Seinfeld

|     | Title              | AirDate   | Writers                                | Director      |   Percent_Positive |   Percent_Negative |
|----:|:-------------------|:----------|:---------------------------------------|:--------------|-------------------:|-------------------:|
|  22 | The Parking Garage | 30-Oct-91 | Larry David                            | Tom Cherones  |           0.209964 |           0.181495 |
|  16 | The Busboy         | 26-Jun-91 | Larry David, Jerry Seinfeld            | Tom Cherones  |           0.210762 |           0.183857 |
| 171 | The Maid           | 30-Apr-98 | Alec Berg, David Mandel, Jeff Schaffer | Andy Ackerman |           0.225275 |           0.17033  |
|  88 | The Chinese Woman  | 13-Oct-94 | Peter Mehlman                          | Andy Ackerman |           0.226415 |           0.18239  |
|  39 | The Keys           | 6-May-92  | Larry Charles                          | Tom Cherones  |           0.240678 |           0.183051 |

![](images/pos_neg_sentiment_season.png)

![](images/pos_neg_sentiment_character.png)


[Back to Top](#Table-of-Contents)


# Text Generation
While I haven't done much with text generation yet, I have been looking into several options for my model, and will most likely use a LSTM RNN. There are quite a few different methodologies that can be applied when creating these types of models, and the two main ways to generate text are character-based, and word-based. In a character-based model, the prediction happens one character at a time, with the neural network inputs being the N characters before the character being predicted. This allows for the number to potential predictions (y-values) to be quite small, in that it's only usually: `letters in the alphabet + digits 0-9 + punctuation`.

Contrast this with trying to predict a word using the N words before that word, and your prediction space is suddenly your entire vocabulary! This makes for a potentially HUGE (30,000+ depending on vocab) # of word prediction options, which will in turn require much more processing power. However, I believe that this type of model "might" be more accurate overall, in that it can internally take the context of each word into account, within the LSTM neural network.


[Back to Top](#Table-of-Contents)

# Reflection and Future Work
## Reflection
- 

## Future Work
- 

[Back to Top](#Table-of-Contents)