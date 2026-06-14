import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def render_distribution_pie(filtered_df):
    """Renders the pie chart showing distribution between datasets."""
    st.subheader("Dataset Record Breakdown")
    if not filtered_df.empty:
        distribution_df = filtered_df['dataset_source'].value_counts().reset_index()
        distribution_df.columns = ['Dataset Source', 'Total Records']
        
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
        st.info("Select a dataset source to plot distributions.")

def render_length_boxplot(filtered_df):
    """Renders the boxplot tracking character count distribution."""
    st.subheader("Length Verification Boxplot")
    if not filtered_df.empty:
        fig_box = px.box(
            filtered_df,
            x='dataset_source',
            y='char_count',
            color='dataset_source',
            labels={'dataset_source': 'Dataset Source', 'char_count': 'Character Count'},
            color_discrete_map={'Training': '#1F77B4', 'Validation': '#6BAED6'}
        )
        st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Filter view is empty.")

def render_wordcloud(filtered_df):
    """Generates and renders the keyword word cloud."""
    st.subheader("Keyword Cloud Map")
    if not filtered_df.empty:
        text_dump = " ".join(filtered_df['tweet_text'].tolist())
        if text_dump.strip():
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