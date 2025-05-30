import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from google.cloud import storage
import os
import uuid
from datetime import datetime
import numpy as np

# --- Configuration ---
GCP_PROJECT_ID = "your-gcp-project-id"
GCP_BUCKET_NAME = "wildlife-sightings-data"
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

# --- GCP Data Upload ---
def upload_to_gcs(df, bucket_name, file_name):
    """Upload DataFrame to Google Cloud Storage as CSV."""
    try:
        client = storage.Client(project=GCP_PROJECT_ID)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(file_name)
        df.to_csv(file_name, index=False)
        blob.upload_from_filename(file_name)
        os.remove(file_name)
        return f"Uploaded {file_name} to gs://{bucket_name}/{file_name}"
    except Exception as e:
        return f"Error uploading to GCS: {str(e)}"

# --- Visualizations ---
def create_scatter_plot(df):
    """Create scatter plot of species vs. count."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='species', y='count', hue='species', size='count', sizes=(50, 200))
    plt.title('Species Sightings vs. Count')
    plt.xlabel('Species')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('scatter_plot.png')
    plt.close()
    return 'scatter_plot.png'

def create_bar_plot(df):
    """Create bar plot of sightings by region."""
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x='region', hue='region')
    plt.title('Sightings by Region')
    plt.xlabel('Region')
    plt.ylabel('Number of Sightings')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('bar_plot.png')
    plt.close()
    return 'bar_plot.png'

def create_heatmap(df):
    """Create heatmap of sightings by hour and region."""
    pivot_table = df.pivot_table(values='count', index='hour', columns='region', aggfunc='sum', fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap='YlOrRd', annot=True, fmt='.0f')
    plt.title('Sightings Frequency by Hour and Region')
    plt.xlabel('Region')
    plt.ylabel('Hour of Day')
    plt.tight_layout()
    plt.savefig('heatmap.png')
    plt.close()
    return 'heatmap.png'

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
        scatter_plot = create_scatter_plot(filtered_df)
        st.image(scatter_plot, caption="Species vs. Count")
        
    with col2:
        bar_plot = create_bar_plot(filtered_df)
        st.image(bar_plot, caption="Sightings by Region")

    st.subheader("Sightings Heatmap")
    heatmap = create_heatmap(filtered_df)
    st.image(heatmap, caption="Sightings by Hour and Region")

    # Upload to GCS
    st.subheader("Upload to Google Cloud Storage")
    if st.button("Upload Data to GCS"):
        file_name = f"sightings_{uuid.uuid4()}.csv"
        result = upload_to_gcs(filtered_df, GCP_BUCKET_NAME, file_name)
        st.write(result)

if __name__ == "__main__":
    main()