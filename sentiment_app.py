import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
import numpy as np

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Airline Sentiment Dashboard",
    page_icon="✈️",
    layout="wide"
)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

@st.cache_data
def load_data():
    """
    Loads the Tweets.csv file. 
    If the file is not found, generates sample data for demonstration.
    """
    try:
        # Try to read the user's file
        df = pd.read_csv("Tweets.csv")
        # Ensure 'tweet_created' is datetime
        if 'tweet_created' in df.columns:
            df['tweet_created'] = pd.to_datetime(df['tweet_created'])
        return df
    except FileNotFoundError:
        st.warning("⚠️ 'Tweets.csv' not found. Using generated sample data for demonstration.")
        # Create dummy data structure matching the user's requirements
        data = {
            'airline_sentiment': np.random.choice(['positive', 'neutral', 'negative'], 100, p=[0.2, 0.3, 0.5]),
            'negativereason': np.random.choice(['Late Flight', 'Lost Luggage', 'Customer Service', 'Cancelled Flight', 'Booking Issue'], 100),
            'airline': np.random.choice(['United', 'Delta', 'US Airways', 'American', 'Southwest', 'Virgin America'], 100),
            'text': [
                "@VirginAmerica What @dhepburn said.", 
                "@United car broke down on way to airport, can I change flight?", 
                "@SouthwestAir lost my bag again!",
                "@USAirways thanks for the great service today.",
                "@JetBlue flight was delayed 3 hours. #frustrated",
                "@AmericanAir worst customer service ever.",
                "@Delta love the new seats!",
                "@United luggage was damaged.",
                "@VirginAmerica you guys rock!",
                "@SouthwestAir on hold for 40 mins..."
            ] * 10,
            'tweet_created': pd.date_range(start='1/1/2023', periods=100)
        }
        df = pd.DataFrame(data)
        # Fix negative reason to be NaN if sentiment is not negative (for realism)
        df.loc[df['airline_sentiment'] != 'negative', 'negativereason'] = np.nan
        return df

def clean_text(text):
    """
    Cleans text by removing mentions (@user), links, and special characters.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'@[A-Za-z0-9]+', '', text) # Remove mentions
    text = re.sub(r'https?://[A-Za-z0-9./]+', '', text) # Remove links
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special chars (keep letters/spaces)
    return text.lower().strip()

# -----------------------------------------------------------------------------
# Main App Logic
# -----------------------------------------------------------------------------

def main():
    st.title("✈️ Airline Customer Sentiment")
    st.markdown("Analyze how travelers feel about different airlines based on Twitter data.")

    # 1. Load Data
    df = load_data()
    
    # Preprocessing: Add cleaned text column
    df['cleaned_text'] = df['text'].apply(clean_text)

    # 2. Sidebar Filters
    st.sidebar.title("Filter Options")
    st.sidebar.markdown("Customize your view:")
    
    # Airline Selection
    airline_options = ["All Airlines"] + sorted(df['airline'].unique().tolist())
    selected_airline = st.sidebar.selectbox("Select Airline", airline_options)

    # Filter Data based on selection
    if selected_airline == "All Airlines":
        filtered_df = df
    else:
        filtered_df = df[df['airline'] == selected_airline]

    # 3. Tabs Layout
    tab1, tab2 = st.tabs(["📊 Dashboard", "📄 Raw Data"])

    # --- TAB 1: DASHBOARD ---
    with tab1:
        # Top KPI Row
        total_tweets = len(filtered_df)
        sentiment_counts = filtered_df['airline_sentiment'].value_counts()
        
        # Safe get for counts
        pos_count = sentiment_counts.get('positive', 0)
        neu_count = sentiment_counts.get('neutral', 0)
        neg_count = sentiment_counts.get('negative', 0)

        # Calculate percentages
        pos_pct = (pos_count / total_tweets * 100) if total_tweets > 0 else 0
        neg_pct = (neg_count / total_tweets * 100) if total_tweets > 0 else 0

        # KPI Columns
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Tweets", f"{total_tweets:,}")
        col2.metric("Positive Sentiment", f"{pos_pct:.1f}%", f"{pos_count} tweets")
        col3.metric("Negative Sentiment", f"{neg_pct:.1f}%", f"{neg_count} tweets", delta_color="inverse")
        
        st.divider()

        # Row 2: Charts (Pie + Bar)
        c1, c2 = st.columns((1, 1))

        with c1:
            st.subheader("Sentiment Distribution")
            if not filtered_df.empty:
                fig_pie = px.pie(
                    filtered_df, 
                    names='airline_sentiment', 
                    values=filtered_df.index, # Use index to count
                    hole=0.4,
                    color='airline_sentiment',
                    color_discrete_map={
                        'positive': '#2ECC71', 
                        'neutral': '#F1C40F', 
                        'negative': '#E74C3C'
                    }
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No data available.")

        with c2:
            st.subheader("Top Negative Reasons")
            
            # Filter specifically for valid negative reasons (remove NaNs and empty strings)
            if 'negativereason' in filtered_df.columns:
                negative_df = filtered_df[
                    (filtered_df['airline_sentiment'] == 'negative') & 
                    (filtered_df['negativereason'].notna()) & 
                    (filtered_df['negativereason'] != "")
                ]
            else:
                negative_df = pd.DataFrame() # Empty if column missing

            if not negative_df.empty:
                reason_counts = negative_df['negativereason'].value_counts().reset_index()
                reason_counts.columns = ['Reason', 'Count']
                
                fig_bar = px.bar(
                    reason_counts, 
                    x='Count', 
                    y='Reason', 
                    orientation='h', 
                    color='Count',
                    color_continuous_scale='Reds'
                )
                fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("No negative reasons found to display.")

        st.divider()

        # Row 3: Word Cloud & List Combined
        st.subheader("📝 Text Analysis: What are people saying?")
        
        negative_text_df = filtered_df[filtered_df['airline_sentiment'] == 'negative']
        
        if not negative_text_df.empty:
            all_text = " ".join(negative_text_df['cleaned_text'])
            
            # --- Extended Stopwords List ---
            stopwords = {
                'flight', 'airline', 'thank', 'thanks', 'plane', 'trip', 'customer', 'service',
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd",
                'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers',
                'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were',
                'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
                'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
                'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
                'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain',
                'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
                'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't",
                'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
                'won', "won't", 'wouldn', "wouldn't", 'get', 'got'
            }

            # Layout: 2 Columns (Cloud vs List)
            wc_col, list_col = st.columns(2)

            # --- Left Column: Word Cloud ---
            with wc_col:
                st.write("**Word Cloud**")
                wc = WordCloud(
                    width=800, 
                    height=500, 
                    background_color='white',
                    colormap='Reds',
                    stopwords=stopwords,
                    min_font_size=10
                ).generate(all_text)
                
                fig_wc, ax = plt.subplots(figsize=(10, 6))
                ax.imshow(wc, interpolation='bilinear')
                ax.axis('off')
                plt.tight_layout(pad=0)
                st.pyplot(fig_wc)
                plt.close(fig_wc)

            # --- Right Column: Frequency List ---
            with list_col:
                st.write("**Top 20 Keywords**")
                
                # Tokenize and filter
                words = all_text.split()
                filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
                
                if filtered_words:
                    word_counts = Counter(filtered_words)
                    common_words_df = pd.DataFrame(word_counts.most_common(20), columns=['Word', 'Frequency'])
                    
                    st.dataframe(
                        common_words_df,
                        use_container_width=True,
                        hide_index=True,
                        height=500, # Match approximate height of the image
                        column_config={
                            "Word": st.column_config.TextColumn("Word"),
                            "Frequency": st.column_config.ProgressColumn(
                                "Frequency",
                                format="%d",
                                min_value=0,
                                max_value=int(common_words_df['Frequency'].max()),
                            ),
                        }
                    )
                else:
                    st.info("Not enough data to generate list.")
        else:
            st.write("Not enough negative text data to analyze.")

    # --- TAB 2: RAW DATA ---
    with tab2:
        st.subheader("Raw Twitter Data")
        st.write(f"Showing data for: **{selected_airline}**")
        
        # Toggle to show clean text
        show_clean = st.checkbox("Show Cleaned Text Column")
        
        display_cols = ['airline_sentiment', 'airline', 'text', 'negativereason', 'tweet_created']
        if show_clean:
            display_cols.append('cleaned_text')
            
        st.dataframe(
            filtered_df[display_cols].sort_values(by='tweet_created', ascending=False),
            use_container_width=True,
            hide_index=True
        )

if __name__ == "__main__":
    main()