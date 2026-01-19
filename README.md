Airline Customer Sentiment Dashboard:

A Streamlit-based web application that analyzes customer sentiment towards major US airlines using Twitter data. The dashboard provides interactive visualizations, key performance indicators (KPIs), and text analysis of customer complaints.

Features:

Interactive Sidebar: Filter data by specific airlines or view aggregate data for all airlines.

Key Metrics: Real-time calculation of Total Tweets, Positive Sentiment %, and Negative Sentiment %.

Visualizations:

Sentiment Distribution: Interactive Pie Chart (Plotly) showing the breakdown of positive, neutral, and negative tweets.

Negative Reasons: Bar chart displaying the most common reasons for complaints (e.g., Late Flight, Lost Luggage).

Text Analysis:

Word Cloud: Visual representation of common terms in negative tweets.

Top Keywords List: Frequency table of the top 20 words used in complaints.

Raw Data Explorer: View and sort the underlying dataset with an option to see cleaned text.

Installation:

Clone the repository:

git clone [https://github.com/Manaskumarparhi/Airline-Customer-Sentiment-Dashboard.git](https://github.com/Manaskumarparhi/Airline-Customer-Sentiment-Dashboard.git)
cd Airline-Customer-Sentiment-Dashboard


Install dependencies:

pip install streamlit pandas plotly wordcloud matplotlib numpy


Run the application:

streamlit run sentiment_app.py


Data Filtering Logic:

To ensure the Text Analysis (Word Cloud and Top Keywords) focuses on meaningful complaints rather than common grammatical words, a comprehensive "stopword" filter is applied.

The following words are excluded from the analysis:

1. Domain-Specific Words

Words that are too generic to the context of aviation and obscure specific issues:

flight, airline, plane, trip

customer, service

thank, thanks

2. General English Stopwords

Common words that add structure but little semantic value to sentiment analysis:

Pronouns: i, me, my, you, your, he, she, it, we, they, them, their, myself, yourself, themselves, etc.

Auxiliary Verbs: is, are, was, were, be, been, have, has, had, do, does, did, can, will, should, could, would, get, got.

Prepositions & Conjunctions: and, but, if, or, because, as, of, at, by, for, with, about, to, from, in, on, off, over, under, between, into, through.

Adverbs & Question Words: now, when, where, why, how, here, there, just, only, very, so, too, again, then.

Negations (handled separately): no, not, nor (and contractions like don't, won't, isn't).

File Structure:

sentiment_app.py: The main Streamlit application script.

Tweets.csv: Input dataset containing airline sentiment data (required).

README.md: Project documentation.

Robustness & Error Handling:

The application implements a graceful fallback mechanism for data loading:

Primary Path: Attempts to load the local Tweets.csv dataset.

Fallback Path: In the event the CSV file is missing or corrupted, the system dynamically generates a synthetic dataset. This ensures the UI components and visualization logic remain testable and demonstrable for reviewers without causing an application crash.
