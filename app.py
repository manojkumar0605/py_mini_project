import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import uuid
import numpy as np
from datetime import datetime

# --- Configuration ---
CSV_FILE_PATH = "sightings_data.csv"

# --- Data Generation (Simulating eBird/iNaturalist-like data) ---
@st.cache_data
def generate_sample_data(n=1000):
    np.random.seed(42)
    species = ['Bald Eagle', 'Red Fox', 'Black Bear', 'Great Horned Owl', 'White-tailed Deer']
    regions = ['North Forest', 'South Plains', 'East Wetlands', 'West Mountains']
    times = pd.date_range(start='2025-01-01', end='2025-05-30', freq='H')
    
    data = {
        'species': np.random.choice(species, n),
        'count': np.random.randint(1, 10, n),
        'region': np.random.choice(regions, n),
        'timestamp': np.random.choice(times, n),
        'latitude': np.random.uniform(35.0, 45.0, n),
        'longitude': np.random.uniform(-120.0, -100.0, n)
    }
    df = pd.DataFrame(data)
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date
    return df

# --- Visualizations ---
@st.cache_data
def create_scatter_plot(df):
    """Create scatter plot of species vs. count."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='species', y='count', hue='species', size='count', sizes=(50, 200))
    plt.title('Species Sightings vs. Count')
    plt.xlabel('Species')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_name = 'scatter_plot.png'
    plt.savefig(file_name)
    plt.close()
    # Read the image into memory and delete the file
    with open(file_name, "rb") as f:
        img_data = f.read()
    os.remove(file_name)
    return img_data

@st.cache_data
def create_bar_plot(df):
    """Create bar plot of sightings by region."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='region', hue='region')
    plt.title('Sightings by Region')
    plt.xlabel('Region')
    plt.ylabel('Number of Sightings')
    plt.xticks(rotation=45)
    plt.tight_layout()
    file_name = 'bar_plot.png'
    plt.savefig(file_name)
    plt.close()
    # Read the image into memory and delete the file
    with open(file_name, "rb") as f:
        img_data = f.read()
    os.remove(file_name)
    return img_data

@st.cache_data
def create_heatmap(df):
    """Create heatmap of sightings by hour and region."""
    pivot_table = df.pivot_table(values='count', index='hour', columns='region', aggfunc='sum', fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='.0f')
    plt.title('Sightings Frequency by Hour and Region')
    plt.xlabel('Region')
    plt.ylabel('Hour of Day')
    plt.tight_layout()
    file_name = 'heatmap.png'
    plt.savefig(file_name)
    plt.close()
    # Read the image into memory and delete the file
    with open(file_name, "rb") as f:
        img_data = f.read()
    os.remove(file_name)
    return img_data

# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Wildlife Sightings Dashboard", layout="wide")
    st.title("Wildlife Sightings Dashboard")
    st.markdown("Track and visualize wildlife sightings across regions.")

    # Load or generate data
    df = generate_sample_data()

    # Sidebar for filters
    st.sidebar.header("Filters")
    selected_species = st.sidebar.multiselect("Select Species", options=df['species'].unique(), default=df['species'].unique())
    selected_regions = st.sidebar.multiselect("Select Regions", options=df['region'].unique(), default=df['region'].unique())
    
    # Filter data
    filtered_df = df[df['species'].isin(selected_species) & df['region'].isin(selected_regions)]

    # Display data
    st.subheader("Sighting Data")
    st.dataframe(filtered_df)

    # Visualizations
    st.subheader("Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        scatter_plot_data = create_scatter_plot(filtered_df)
        st.image(scatter_plot_data, caption="Species vs. Count")
        
    with col2:
        bar_plot_data = create_bar_plot(filtered_df)
        st.image(bar_plot_data, caption="Sightings by Region")

    st.subheader("Sightings Heatmap")
    heatmap_data = create_heatmap(filtered_df)
    st.image(heatmap_data, caption="Sightings by Hour and Region")

    # Download filtered data as CSV
    st.subheader("Download Filtered Data")
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Data as CSV",
        data=csv,
        file_name=f"sightings_{uuid.uuid4()}.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
