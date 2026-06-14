import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# Page Configuration & Header Setup
st.set_page_config(
    page_title="Twitter Content Analysis Portal",
    page_icon="📊",
    layout="wide"
)
st.title("Twitter Content EDA Dashboard")
st.markdown("### SAP ID:  70176505 | Student Assignment Project")
st.write("An all-in-one exploratory data analysis dashboard tracking text volume, sentiment distributions, and entity metrics.")
st.markdown("---")

# Data Loading Strategy using Dynamic OS Path Lookup & Secure Parsing
@st.cache_data
def load_combined_data():
    # Standardized header names matching your actual CSV dataset columns
    cols = ['id', 'entity', 'sentiment', 'tweet_text']
    
    # Dynamically find the absolute path of the folder containing this script file
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Construct paths checking for common filename extensions present in your workspace
    training_path = os.path.join(BASE_DIR, "twitter_training.csv.txt")
    validation_path = os.path.join(BASE_DIR, "twitter_validation.csv.txt")
    
    # Fallback to standard .csv naming conventions if .txt forms aren't found
    if not os.path.exists(training_path):
        training_path = os.path.join(BASE_DIR, "twitter_training.csv")
        validation_path = os.path.join(BASE_DIR, "twitter_validation.csv")
    
    # Fallback to absolute strings if folder structure uses original "twitter_" naming conventions
    if not os.path.exists(training_path):
        training_path = os.path.join(BASE_DIR, "twitter_training.csv")
        validation_path = os.path.join(BASE_DIR, "twitter_validation.csv")
        
    try:
        # Load Training Data using engine='python' to safely isolate text commas
        df_train = pd.read_csv(
            training_path, 
            names=cols, 
            header=None, 
            on_bad_lines='skip', 
            engine='python'
        )
        df_train['dataset_source'] = 'Training'
        
        # Load Validation Data using engine='python' to safely isolate text commas
        df_val = pd.read_csv(
            validation_path, 
            names=cols, 
            header=None, 
            on_bad_lines='skip', 
            engine='python'
        )
        df_val['dataset_source'] = 'Validation'
        
        # Merge both datasets cleanly
        df = pd.concat([df_train, df_val], ignore_index=True)
        
        # Clean columns securely handling empty rows
        df['tweet_text'] = df['tweet_text'].fillna("").astype(str).str.strip()
        df['entity'] = df['entity'].fillna("Unknown").astype(str).str.strip()
        df['sentiment'] = df['sentiment'].fillna("Neutral").astype(str).str.strip()
        
        # Feature Engineering: Generate text lengths for metrics
        df['char_count'] = df['tweet_text'].apply(len)
        df['word_count'] = df['tweet_text'].apply(lambda x: len(x.split()))
        
        return df
    except Exception as e:
        st.error(f"Failed to load datasets. Please verify your files are in the same folder as this script. Error details: {e}")
        st.stop()

# Initialize data
df = load_combined_data()

# ==================== FILTER PANEL (Sidebar Controls) ====================
st.sidebar.header("Dashboard Control Panel")

# 1. Interactive Dataset Filter (Train vs Validation toggle)
source_options = df['dataset_source'].unique()
selected_sources = st.sidebar.multiselect(
    "Filter by Dataset Source:",
    options=source_options,
    default=list(source_options)
)

# 2. Interactive Entity Filter (Games, Brands, etc.)
entity_options = sorted(df['entity'].unique())
selected_entities = st.sidebar.multiselect(
    "Filter by Entity/Game:",
    options=entity_options,
    default=entity_options[:5] if len(entity_options) > 5 else entity_options  # Default to first 5 items to keep screen clean
)

# 3. Interactive Sentiment Filter
sentiment_options = df['sentiment'].unique()
selected_sentiments = st.sidebar.multiselect(
    "Filter by Sentiment Profiles:",
    options=sentiment_options,
    default=list(sentiment_options)
)

# 4. Interactive Keyword Search Slicer
search_query = st.sidebar.text_input("Search Specific Keyword in Tweets:", "")

# Apply runtime dataframe scoping instantly based on control picks
filtered_df = df[
    df['dataset_source'].isin(selected_sources) & 
    df['entity'].isin(selected_entities) & 
    df['sentiment'].isin(selected_sentiments)
]

if search_query:
    filtered_df = filtered_df[filtered_df['tweet_text'].str.contains(search_query, case=False, na=False)]

# ==================== MAIN CHARTS & VISUALS SECTION ====================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Dataset Record Breakdown")
    if not filtered_df.empty:
        distribution_df = filtered_df['dataset_source'].value_counts().reset_index()
        distribution_df.columns = ['Dataset Source', 'Total Records']
        
        # Plotly pie chart definition using professional Blue palette
        fig_pie = px.pie(
            distribution_df,
            values='Total Records',
            names='Dataset Source',
            hole=0.4,
            color='Dataset Source',
            color_discrete_map={'Training': '#1F77B4', 'Validation': '#6BAED6'}
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("Select options in the sidebar control panel to plot distributions.")

with col2:
    st.subheader("Sentiment Distribution Profile")
    if not filtered_df.empty:
        # Plotly grouped bar chart tracking sentiments across your chosen parameters
        sentiment_df = filtered_df.groupby(['dataset_source', 'sentiment']).size().reset_index(name='Counts')
        fig_bar = px.bar(
            sentiment_df,
            x='sentiment',
            y='Counts',
            color='dataset_source',
            barmode='group',
            labels={'sentiment': 'Sentiment Classification', 'Counts': 'Total Tweets'},
            color_discrete_map={'Training': '#1F77B4', 'Validation': '#6BAED6'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Filter view is empty.")

st.markdown("---")

# ==================== TEXT METRICS & REVIEWS ====================
col3, col4 = st.columns(2)

with col3:
    st.subheader("Keyword Cloud Map")
    if not filtered_df.empty:
        text_dump = " ".join(filtered_df['tweet_text'].tolist())
        if text_dump.strip():
            # Configured WordCloud to use the built-in 'Blues' color profile
            wordcloud = WordCloud(
                width=600,
                height=350,
                background_color='white',
                colormap='Blues'
            ).generate(text_dump)
            
            fig_wc, ax = plt.subplots(figsize=(6, 3.5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig_wc)
        else:
            st.info("No text content available to generate a word cloud.")
    else:
        st.info("Filter view is empty.")

with col4:
    st.subheader("Tabular Dataset Previewer")
    if not filtered_df.empty:
        # Show clean columns matching your updated data criteria
        st.dataframe(
            filtered_df[['id', 'dataset_source', 'entity', 'sentiment', 'char_count', 'word_count', 'tweet_text']],
            use_container_width=True, 
            height=280
        )
    else:
        st.info("No records match the active filter criteria.")