# Sentiment Analysis
Sentiment analysis is a very popular task in natural language processing (NLP). It belongs to a subtask or application of text classification, where sentiments or subjective information from different texts are extracted and identified. Today, many businesses around the world use sentiment analysis to understand more deeply their customers and clients by analyzing sentiments across different target groups. It also has wide applications in different sources of information, including product reviews, online social media, survey feedback, etc.


## Datasets
I have performed sentiment analysis on 2 datasets i.e. Amazon reviews and IMDB dataset. Amazon dataset has a list of over 39,000 consumer reviews for Amazon products like the Kindle, Fire TV Stick, and more provided by Datafiniti's Product Database. The dataset includes basic product information, rating, review text, and more for each product. IMDB dataset by Keras is a dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). Reviews have been preprocessed, and each review is encoded as a list of word indexes (integers). For convenience, words are indexed by overall frequency in the dataset, so that for instance the integer "3" encodes the 3rd most frequent word in the data.

## Exploratory Data Analysis (EDA)
Since we are interested in sentiment analysis, we will only use review and its rating in amazon dataset. The rating >=4 is positive and <=3 is negative. The significant aspect is to import libraries for our use statistical analysis cases. Some of these include:
- Sklearn
- Matplotlib
- Seaborn
- NLTK
- wordcloud 

These libraries and frameworks are efficient in handling data which can be used for data and result analysis.

<img src="https://github.com/Muskan-goyal6/SentimentAnalysis/blob/master/images/Screenshot%202021-07-25%20at%202.14.49%20PM.png"/>
<img src="https://github.com/Muskan-goyal6/SentimentAnalysis/blob/master/images/Screenshot%202021-07-25%20at%202.15.22%20PM.png", width=400,height=400/>
<img src="https://github.com/Muskan-goyal6/SentimentAnalysis/blob/master/images/Screenshot%202021-07-25%20at%202.15.56%20PM.png", width=460, height=400/>
<img src="https://github.com/Muskan-goyal6/SentimentAnalysis/blob/master/images/Screenshot%202021-07-25%20at%202.30.58%20PM.png", width=460, height=400/>
<img src="https://github.com/Muskan-goyal6/SentimentAnalysis/blob/master/images/Screenshot%202021-07-25%20at%205.15.48%20PM.png", width=450, height=450/>


## Data cleaning and pre-processing
There is a presence of certain html tags and punctuations which have to be removed as these are adding noise to the Amazon review corpus. This will be taken up in the cleaning phase. We also remove the redundancies like HTML codes, URLs, Emojis, Stopwords, Punctuations, Abbreviations. These will be sufficient for cleaning the corpus.  

Transforming the Corpus!! Now at this stage the data is successfully cleaned and all redundant noises are removed. These steps are generic to any NLP pipeline which reduces the dimension of the data. Once the data is cleaned , we can again prune some words to their base form and reduce the sentence lengths. This is important because when we are applying any model (statistical, deep learning, transformers,graphs), 2 different words from the same base word are encoded and tokenized in a different manner. For instance, the word "watched" and "watching" have the same root word "watch", however they are encoded separately with respect to any Tokenizer.

To alleviate this issue, it is recommended to perform lemmatization on the text corpus so that the words can be reduced to their root semantic word. Morphological transformations such as "watched" and "watching", are converted to their base form through this method. Stemming , although can be used , is not recommended as it does not take into consideration the semantics of the sentence or the surrounding words which are present around it. Stemming also produces words which are not present in the vocabulary.

So the following are the final steps:
- Clean text
- Remove Stopwords: There can be some words in our sentences that occur very frequently and don't contribute too much to the overall meaning of the sentences. We usually have a list of these words and remove them from each our sentences. For example: "a", "an", "the", "this", "that", "is", "it", "to", "and" in this example.
- Stemming: Stemming is a rule-based system to convert words into their root form. It removes suffixes from words. This helps us enhace similarities (if any) between sentences.
- Lemmatization: If we are not satisfied with the result of stemming, we can use the Lemmatization instead. It usually requires more work, but gives better results. As mentioned in the class, lemmatization needs to know the correct word position tags such as "noun", "verb", "adjective", etc. and we will use another NLTK function to feed this information to the lemmatizer.

## Importance of data cleaning and pre-processing
The afore mentioned phase is one of the most important phase. If the textual data is not properly cleaned or processed, incorrect words/puncutations/urls and associated redundancies get added to the data. This impacts the performance when we will be creating static/dynamic embeddings and analysing the sentence/word vectors. In the context of embeddings,(and subsequently models), we will find that if we donot remove these inconsistencies, the vectors will not be properly placed. 

## Feature Engineering and Selection
- Create a vectorizer (TF-IDF or count)
- Splitting Dataset into Train and Test Set
- Oversampling

## Models Used
- Naive Bayes
- XGBoost
- small MLP
- simple RNN model
- pretrained BERT : We will be using pretrained transformers rather than fine-tuning our own, so a low setup cost is needed.

## Results

| Model      | IMDB | AMAZON     |
|   :----:    |   :----:    |   :----:    |
| Naive Bayes| 84%       | 80%  |
| XGBoost   | 86%        | 81%      |
| MLP   | 88%       | 91.56%     |
| RNN   | 84%       | 93%     |
| BERT   | 89%       | 95%     |





