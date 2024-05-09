"""
Download datasets from Cornell Conversational Analysis Toolkit
"""

from convokit import Corpus, download

movie_corpus = Corpus(filename=download("movie-corpus"))
movie_corpus.print_summary_stats()

subreddit_corpus = Corpus(filename=download("subreddit-Cornell"))
subreddit_corpus.print_summary_stats()
