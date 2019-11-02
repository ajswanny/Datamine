# Datamine
Datamine is a project that was inspired by the desire to gain some knowledge about user-created content on the website *Reddit* The project is composed of a set of programs that form a data pipeline...

#### *DataCollector*
Grabs comment data from a specified Reddit *subreddit* and stores it as JSON data.

#### *DataProcessor*
Compiles raw data to [Pandas](https://pandas.pydata.org) data structures for analysis or observation.

#### *DataCleaner*
A set of functions to clean subreddit data compiled into Pandas data structures.

#### *DataObserver*
A class of functions for data observation and analysis with functionality to analyze natural language sentiment.

#### Work done
Although this was a very amateurish attempt at data science, some interesting results were obtained. Below are just a couple of the many results of sentiment analysis of Reddit comments in the *r/news* subreddit.

|Comment text|Text category|Date created|Sentiment score|
|------------|--------|------------|---------------|
|One of the most interesting sentences to me, a...|People & Society|2015-06-26|0.6|
|The whole "he'll set himself on fire" hooplah ...|Arts & Entertainment|2015-06-26|-0.4|
