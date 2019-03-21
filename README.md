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
Seinfeld is an American television sitcom that ran for nine seasons on NBC, from 1989 to 1998. It was created by Larry David and Jerry Seinfeld, with the latter starring as a fictionalized version of himself. Set predominantly in an apartment building in Manhattan's Upper West Side in New York City, the show features a handful of Jerry's friends and acquaintances, including best friend George Costanza, friend and former girlfriend Elaine Benes, and neighbor across the hall Cosmo Kramer. It is often described as being "a show about nothing", as many of its episodes are about the minutiae of daily life.

![](images/nothing.jpg)


I have been a huge fan of this show for around two decades, and still find myself going back and enjoying old episodes. I think the writing is absolutely brilliant, still holds up (even if some references are dated), and really appreciate the way that Larry David in particular can transform seemingly mundane daily occurrences into entertaining dialogue. 

I discovered a database containing scripts for every Seinfeld episode (located [here](https://www.kaggle.com/thec03u5/seinfeld-chronicles)), with dialogue separated by character. Since we have I have not had much previous experience with NLP, and because NLP has so many practical applications, I thought this would be a fun corpus to take on, and try to extract meaning from. 

Initially, I deiced to use LDA and sentiment analysis to better inform my EDA and understanding of the corpus. Unsupervised learning in the form of LDA can quickly turn into an almost endless feedback loop (which I soon realized), but I was able to at least extract some interesting themes, dialogue snippets, and catch phrases from than analysis.

I also thought it would be a very interesting problem to try and see if I can train a model to emulate the speech patterns of the four main characters mentioned above, and even create a model that would “predict”/generate the script for a new episode of Seinfeld.


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
To begin looking into this dataset, I decided to look at who speaks the most, over the entire series. As expected, Jerry has the most lines of all the main characters, followed by George, Kramer and Elaine. I thought it was interesting how large of a drop off there way between these four main characters, and the rest of the characters in the series. Also, while not shown in the graph below, there are ~(FILL THIS IN) unique characters in all seasons combined (way too many to show on this plot).

![](images/lines_by_char_series.png)

I also thought it would be interesting to see how the dialogue was divided among the characters as the seasons progressed, and the chart below shows the % of lines spoken by each main character by season. It's definitely interesting how big of a role Jerry played in the first two seasons, and how that usage dropped quite dramatically from the first through the third season. As I explain below, I believe this might be in part due to the fact that Jerry didn't write as many episodes after the first two seasons. We can also see Kramer's role steadily increasing as the show progresses.

![](images/lines_by_char_season.png)

Another interesting metric that I looked at was who the writers were for each episode. In the plot below I aggregated, % of episodes written across seasons, and we can see the huge role that Larry David and Jerry had in writing the first two seasons. As that writing role diminishes, we can see other writers coming into the fold, but no one writer dominates much after season 5 or so. Larry Charles is another name that gets associated with Seinfeld quite often, and here you can see is increased role in writing seasons two and four.

![](images/writers_by_season.png)

The chart below is meant to illustrate the **massive** amount of different writers and writing teams that contributed to Seinfeld. While we can see that Larry David and Jerry wrote the bulk of the episodes, this definitely shows how much of a collaborative writing effort the series as a whole really was. I also found it fascinating that most of the writing teams only wrote 1 episode, so the fact that the show has such coherence between episodes/characters is interesting. 

![](images/writers_series.png)

The final metrics I decided to look at for my EDA were average number of lines per episode and average number of characters (letters/numbers, not characters in the show) per episode. In the plot below, we can see that average lines per episode stays constant in seasons 1 and 2, and then increases by season ~50 lines per episode in season 3. In general, this metric keeps rising as the seasons go on, and while the episode lengths in terms of minutes stayed the same, this indicates more back-and-forth dialogue between characters. Said differently, this metric shows how many times the person speaking changes, on average, per episode. 

The other metric this graph shows is average characters, or total script length per episode. It's interesting to see the decreasing trend of total script length in the first three seasons, as it seems like in those early seasons each character would speak longer per line of dialogue. After season 3, it seems like the show found a nice balance of lines per episode and script length, as those two appear to be linearly correlated for the rest of the seasons.

![](images/avg_dialogue_chars.png)

## EDA Takeaways
- Jerry definitely speaks the most, as expected, but his lines of dialogue as a % of the total decreased as the seasons went on.
- George's role also decreased a bit after season 4, which corresponds to Larry David writing fewer episodes, and this makes sense because the character of George is based on Larry David himself.
- Larry David and Jerry Seinfeld wrote almost the entire first season themselves, and the bulk of the second season as well.
- After season 4, the writing was done by a much larger pool of people, and no one writer really dominated.
- There are are over (FILL IN) unique writing teams who wrote Seinfeld episodes, but the bulk of them only wrote one episode.
- Lines of dialogue per episode increased as the seasons went on, indicating more back-and-forth dialogue between characters. 


[Back to Top](#Table-of-Contents)

# Unsupervised Learning: LDA with Sklearn
Latent Dirichlet Allocation is an unsupervised modeling technique, used to derive latent topics from corpuses of text (collections of documents). There are many examples of real-world use cases for this technique, such as search engines, text to speech, classifying social media users, and many more.

As with any unsupervised modeling technique, as there is nothing we are really "predicting" with this approach, it is quite difficult to accurately evaluate an LDA model quantitatively. Much of the value gained from topic modeling, and LDA specifically, is the ability to come up with a human-comprehensible understanding of the topics the model spits out.

While optimally this topic labelling can be done by looking at the most important keywords for each topic, it can still be quite difficult to separate topics into concrete "buckets".

I began topic modeling of this corpus by combining the lines of dialogue into one large block of text for each episode, therefore creating a "script" for each episode and each season. I then took the following steps to clean the data and feed it through sklearn's LDA model:
- Corpus = dialogue for each episode
- Remove all apostrophes from each word in corpus since CountVectorizer will not include contractions otherwise.
- Convert corpus to lower case.
- Initially, use NLTK's default stop words.
- Use `CountVectorizer` to convert coprus into term frequency matrix for each document.
- Choose num_topics
- Create lda model using `LatentDirichletAllocation`
- Look **manually** at most important key words for each topic and determine of those **make sense**. 
- If words are repeated often between topics, we're not seeing much differentiation.
- Calculate perplexity or coherence.
- Adjust stop words accordingly and repeat all steps above until you have an LDA model that is somewhat informative.

After doing all of the steps above, I ended up with 10 topics, and the word could below shows the 10 most important keywords for each of those topics. These 10 topics were chosen because with only around 5 or 6 topics, I really wasn't seeing any differentiation between the keywords in each topic, and with more than 10 topics, due to the fact that the corpus only contains 174 episodes, I was seeing some topics that almost encompass only one episode/document.

Also, while this analysis didn't really produce any concrete "topics" that are generalizable to multiple episodes, it did pull out some key words and characters that are very episode-specific. For example, "Keith Hernandez" shows up in topic two, as does "latex", and in the episode "The Boyfriend", Jerry meets his idol, Keith Hernandez (former New York Mets baseball player), and George tells the unemployment office that he's close to a job with Vandelay Industries, a company he made up that makes latex products.

![](images/lda_sklearn_wordcloud.png)


# Unsupervised Learning: LDA with Gensim/spaCy
In order to continue learning as much as possible about LDA, and the python libraries available for NLP, I decided to use Gensim/Spacy to do topic modeling on the corpus of episodes as well. Gensim makes it very easy to create bigram and trigram models, and spaCy's lemmitization feature allows one to take only the parts of speech they are interested in. In this case, I decided to only use nouns, adjectives, verbs, and adverbs, in order to reduce the amount of words that would be less useful to differentiate topics. 

For this model, I also went through many iterations of adding to the stop words list, and these are the additional stop words I used, in order to see more differentiation between the topics:

```
['people', 'happen', 'bad', 'ask', 'anything', 'love' 'nice', 'show','doctor', 'eat', 'hear', 'watch','big' 'meet', 'dog', 'life', 'great', 'kind', 'start', 'funny', 'car', 'keep', 'head', 'find', 'feel' 'everything', 'pick', 'remember', 'boy', 'listen', 'hand', 'sit', 'move', 'sure', 'name', 'still', 'stop', 'wanna', 'new', 'day', 'phone', 'laugh', 'may', 'from', 'subject', 're', 'edu', 'use', 'be', 'get', 'go', 's', 'know', 'see', 'come', 'want', 'look', 'jerry', 'george', 'kramer', 'well', 'tell', 'say', 'think', 'make', 'would', 'could', 'right', 'take', 'good', 'really', 'elaine', 'ill', 'back', 'guy', 'talk', 'something', 'mean', 'thing', 'call', 'give', 'let', 'man', 'little', 'way', 'friend', 'put', 'like', 'time', 'never', 'thank', 'work', 'need', 'woman', 'leave', 'maybe', 'try', 'nothing', 'much'] 
```
As we can see below, this model didn't really perform any better in terms of grouping Seinfeld episodes into "topics", but once again we have quite a few episode specific words and characters, including "yada_yada", "festivus", "fusilli", and "sponge".

![](images/lda_gensim_wordcloud.png)

In order to have a more quantitative approach to evaluating an LDA model, I decided to focus on Gensim's "Coherence" score, which is basically measure of how well a topic model splits documents into easily definable topics. The plot below shows hoe the coherence score changes as the number of topics increases.

![](images/coherence.png)


From the chart above, it appeared that 14 topics resulted in a good balance between number of topics and coherence score, as coherence score didn't increase much after that.

As such, I then re-ran the gensim LDA model with 14 topics, and the gif below is a two-dimensional representation of those topics, along with the 30 most important words for each topic. As we can see, the bulk of the language used within Seinfeld episodes overlaps considerably, with the episodes that are quite different from the norm being shown as smaller topics that are further away from the first 5.

![](images/lda_gif.gif)


# Sentiment Analysis
![](images/problem.gif)

In order to understand this corpus better, and to be able to learn yet another NLP tool, I decided to use NLTK's VADER library to do sentiment analysis on each line of dialogue within the corpus.

I also think that this sentiment analysis could be useful when evaluating the results of my generative model, and could even incorporate some sort of feedback loop to increase the positivity or negativity of a character's dialogue.

VADER is (FILL THIS IN).

Example of a line with positive sentiment:
```

```

Example of a line with negative sentiment:
```


```



The sentiment for the entire series is as follows:
- Positive lines: 31%
- Negative lines: 16%
- Neutral lines: 53%

The plot below shows how sentiment changes over the seasons. We can see that season two by far has largest % of negative lines, and season 6 is the most positive. I thought it was interesting that the number of negative lines seemed to trend with the number of episodes Larry David wrote.

![](images/pos_neg_sentiment_season.png)

The chart below shows how each main character's sentiment changes over the seasons. As we can see, George gets very negative in season two, and then he an all the other characters have less negative lines as time goes on. We can also see that Kramer gets more and more positive as the seasons go on. 
![](images/pos_neg_sentiment_character.png)


##  Episodes with the Highest Percentage of Negative Lines

|   Season - ADD  | Title             | AirDate   | Writers                     | Director      |   Percent Positive |   Percent Negative |
|----:|:------------------|:----------|:----------------------------|:--------------|-------------------:|-------------------:|
|   5 | The Ex-Girlfriend | 16-Jan-91 | Larry David, Jerry Seinfeld | Tom Cherones  |           0.32 |           0.25  |
|   6 | The Pony Remark   | 30-Jan-91 | Larry David, Jerry Seinfeld | Tom Cherones  |           0.28     |           0.24     |
| 134 | The Little Kicks  | 10-Oct-96 | Spike Feresten              | Andy Ackerman |           0.29 |           0.23 |
|  14 | The Baby Shower   | 16-May-91 | Larry Charles               | Tom Cherones  |           0.24 |           0.23  |
|   7 | The Jacket        | 6-Feb-91  | Larry David, Jerry Seinfeld | Tom Cherones  |           0.29 |           0.23 |

## Episodes with Lowest Percentage of Negative lines

|     | Title                                     | AirDate   | Writers                                    | Director           |   Percent Positive |   Percent Negative |
|----:|:------------------------------------------|:----------|:-------------------------------------------|:-------------------|-------------------:|-------------------:|
|  93 | The Secretary                             | 8-Dec-94  | Carol Leifer, Marjorie Gross               | David Owen Trainor |           0.41 |          0.10 |
|  77 | The Marine Biologist                      | 10-Feb-94 | Ron Hague, Charlie Rubin                   | Tom Cherones       |           0.40 |          0.10 |
| 164 | The Reverse Peephole (a.k.a. The Man Fur) | 15-Jan-98 | Spike Feresten                             | Andy Ackerman      |           0.33 |          0.10  |
|  73 | The Cigar Store Indian                    | 9-Dec-93  | Tom Gammill, Max Pross                     | Tom Cherones       |           0.33 |          0.11  |
|  84 | The Opposite                              | 19-May-94 | Andy Cowan and Larry David, Jerry Seinfeld | Tom Cherones       |           0.35 |          0.11  |

## Episodes with Highest Percentage of Positive lines

|    | Title               | AirDate   | Writers                      | Director           |   Percent Positive |   Percent Negative |
|---:|:--------------------|:----------|:-----------------------------|:-------------------|-------------------:|-------------------:|
| 40 | The Trip (1)        | 12-Aug-92 | Larry Charles                | Tom Cherones       |           0.42 |          0.12  |
|  0 | Good News, Bad News | 5-Jul-89  | Larry David, Jerry Seinfeld  | Art Wolff          |           0.42 |          0.15  |
| 82 | The Fire            | 5-May-94  | Larry Charles                | Tom Cherones       |           0.41 |          0.15   |
| 54 | The Visa            | 27-Jan-93 | Peter Mehlman                | Tom Cherones       |           0.41 |          0.17  |
| 93 | The Secretary       | 8-Dec-94  | Carol Leifer, Marjorie Gross | David Owen Trainor |           0.41 |          0.10 |

## Episodes with Lowest Percentage of Positive lines

|     | Title              | AirDate   | Writers                                | Director      |   Percent_Positive |   Percent_Negative |
|----:|:-------------------|:----------|:---------------------------------------|:--------------|-------------------:|-------------------:|
|  22 | The Parking Garage | 30-Oct-91 | Larry David                            | Tom Cherones  |           0.21 |           0.18 |
|  16 | The Busboy         | 26-Jun-91 | Larry David, Jerry Seinfeld            | Tom Cherones  |           0.21 |           0.18 |
| 171 | The Maid           | 30-Apr-98 | Alec Berg, David Mandel, Jeff Schaffer | Andy Ackerman |           0.23 |           0.17  |
|  88 | The Chinese Woman  | 13-Oct-94 | Peter Mehlman                          | Andy Ackerman |           0.23 |           0.18  |
|  39 | The Keys           | 6-May-92  | Larry Charles                          | Tom Cherones  |           0.24 |           0.18 |


## Tying this all together
In order to cohesively combine all of the above analysis, I thought it'd be fun to take a look at one of my favorite episodes, and see how it looks with respect to topic modeling and sentiment analysis. 

Epside name: **"The Summer of George"**

|     | Title              | AirDate   | Writers                                | Director      |   Percent_Positive |   Percent_Negative |
|----:|:-------------------|:----------|:---------------------------------------|:--------------|-------------------:|-------------------:|
|Sklearn Topic|5 |:----------|:---------------------------------------|:--------------|-------------------:|-------------------:|



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


```
jerry: hi, how's your day going today, what is the deal?

george: yeah. 

george: i dont know. its not easy so up. 

jerry: i dont have to eat. 

george: i gotta go to the lost battle. 

jerry: why does he do t
```