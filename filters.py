import streamlit as st

def render_sidebar_filters(df):
    """
    Renders the dashboard control panel sidebar filters 
    and returns the scoped dataframe.
    """
    st.sidebar.header("Dashboard Control Panel")

    # 1. Interactive Dataset Filter (Train vs Validation toggle)
    source_options = df['dataset_source'].unique()
    selected_sources = st.sidebar.multiselect(
        "Filter by Dataset Source:",
        options=source_options,
        default=list(source_options)
    )

    # 2. Interactive Keyword Search Slicer
    search_query = st.sidebar.text_input("Search Specific Keyword in Tweets:", "")

    # Apply runtime dataframe scoping instantly
    filtered_df = df[df['dataset_source'].isin(selected_sources)]
    
    if search_query:
        filtered_df = filtered_df[filtered_df['tweet_text'].str.contains(search_query, case=False, na=False)]
        
    return filtered_df