import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import folium
from streamlit_folium import st_folium

# Page configuration
st.set_page_config(page_title="Sacred Temples of India", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for temple-inspired aesthetic
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Cormorant+Garamond:wght@300;400;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a0a00 50%, #0a0a0a 100%);
    }
    
    .main-title {
        font-family: 'Cinzel', serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #d4af37 0%, #f4e4c1 50%, #d4af37 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 2rem 0;
        text-shadow: 0 0 30px rgba(212, 175, 55, 0.3);
        letter-spacing: 0.1em;
    }
    
    .subtitle {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.3rem;
        text-align: center;
        color: #c9b896;
        margin-bottom: 3rem;
        font-weight: 300;
        letter-spacing: 0.05em;
    }
    
    .section-header {
        font-family: 'Cinzel', serif;
        font-size: 2rem;
        color: #d4af37;
        margin: 2rem 0 1.5rem 0;
        text-align: center;
        border-bottom: 2px solid rgba(212, 175, 55, 0.3);
        padding-bottom: 1rem;
        letter-spacing: 0.08em;
    }
    
    .temple-card {
        background: linear-gradient(135deg, rgba(20, 10, 0, 0.9) 0%, rgba(40, 20, 5, 0.95) 100%);
        border: 2px solid rgba(212, 175, 55, 0.6);
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(212, 175, 55, 0.2), inset 0 0 20px rgba(212, 175, 55, 0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .temple-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(212, 175, 55, 0.4), inset 0 0 30px rgba(212, 175, 55, 0.15);
        border-color: rgba(212, 175, 55, 0.9);
    }
    
    .temple-name {
        font-family: 'Cinzel', serif;
        font-size: 1.5rem;
        color: #f4e4c1;
        margin-bottom: 1rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    
    .temple-description {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.1rem;
        color: #c9b896;
        line-height: 1.8;
        margin-bottom: 0.8rem;
        font-weight: 400;
    }
    
    .temple-coordinates {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1rem;
        color: #a89060;
        font-style: italic;
    }
    
    .stTextArea textarea {
        background: rgba(20, 10, 0, 0.8) !important;
        border: 2px solid rgba(212, 175, 55, 0.5) !important;
        border-radius: 8px !important;
        color: #f4e4c1 !important;
        font-family: 'Cormorant Garamond', serif !important;
        font-size: 1.1rem !important;
        padding: 1rem !important;
    }
    
    .stTextArea textarea:focus {
        border-color: rgba(212, 175, 55, 1) !important;
        box-shadow: 0 0 20px rgba(212, 175, 55, 0.4) !important;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #d4af37 0%, #8b7355 100%) !important;
        color: #1a1410 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 2.5rem !important;
        font-family: 'Cinzel', serif !important;
        font-size: 1rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.1em !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.3) !important;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #f4e4c1 0%, #d4af37 100%) !important;
        box-shadow: 0 6px 25px rgba(212, 175, 55, 0.5) !important;
        transform: translateY(-2px) !important;
    }
    
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(212, 175, 55, 0.5), transparent);
        margin: 2rem 0;
    }
    
    .label-text {
        font-family: 'Cinzel', serif !important;
        color: #d4af37 !important;
        font-size: 1.1rem !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: 0.05em !important;
    }
</style>
""", unsafe_allow_html=True)

# Load dataset with caching
@st.cache_data
def load_data():
    df = pd.read_csv('NewAncientTemples.csv')
    return df

df = load_data()

# Fill NaN values in text fields with empty strings
df['Description'] = df['Description'].fillna('')
df['Coordinates'] = df['Coordinates'].fillna('(0, 0)')

# Combine Description for content-based filtering
df['content'] = df['Description']

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Apply TF-IDF transformation
tfidf_matrix = tfidf.fit_transform(df['content'])

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Define recommendation function
def get_recommendations(description=None):
    recommendations = pd.DataFrame()
    
    if description:
        # Filter based on description similarity
        desc_sim = cosine_similarity(tfidf.transform([description]), tfidf_matrix)
        desc_scores = list(enumerate(desc_sim[0]))
        desc_scores = [score for score in desc_scores if score[1] > 0.0]
        desc_scores = sorted(desc_scores, key=lambda x: x[1], reverse=True)
        desc_scores = desc_scores[:10]
        
        desc_indices = [i[0] for i in desc_scores]
        recommendations = df.iloc[desc_indices][['templeName', 'Coordinates', 'Description']]
        
        # Additionally, filter by temple name if similar
        similar_names = df[df['templeName'].str.contains(description, case=False, na=False)]
        recommendations = pd.concat([recommendations, similar_names])
        
        # Remove duplicate recommendations based on templeName
        recommendations = recommendations.drop_duplicates(subset='templeName')
    
    return recommendations

# Function to parse coordinates
def parse_coordinates(coord_str):
    try:
        coords = eval(coord_str)
        if isinstance(coords, (list, tuple)) and len(coords) == 2:
            return [float(coords[0]), float(coords[1])]
    except:
        pass
    return [np.nan, np.nan]

# Static dataset for top places
static_top_places = pd.DataFrame({
    'TempleName': [
        'Temples of Varanasi',
        'Taj Mahal',
        'Golden Temple',
        'Temple of Somnath',
        'Meenakshi Temple'
    ],
    'Description': [
        'Ancient temples in Varanasi, known for their historical significance and spirituality.',
        'Iconic mausoleum located in Agra, one of the Seven Wonders of the World.',
        'Famous Sikh temple located in Amritsar, known for its stunning architecture and holy significance.',
        'Historical Hindu temple dedicated to Lord Shiva, located in Gujarat.',
        'Historic temple in Madurai dedicated to the goddess Meenakshi, famous for its intricate sculptures.'
    ],
    'Coordinates': [
        '(25.3176, 82.9739)',
        '(27.1751, 78.0421)',
        '(31.6200, 74.8760)',
        '(20.8978, 70.0097)',
        '(9.9196, 78.1198)'
    ]
})

# Initialize session states
if 'show_map' not in st.session_state:
    st.session_state.show_map = False
if 'show_recommendations' not in st.session_state:
    st.session_state.show_recommendations = False
if 'show_top_places' not in st.session_state:
    st.session_state.show_top_places = False

# Main title
st.markdown('<h1 class="main-title">SACRED TEMPLES OF INDIA</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Discover the spiritual heritage and architectural wonders of ancient India</p>', unsafe_allow_html=True)

# User input for description
st.markdown('<p class="label-text">Describe Your Spiritual Journey</p>', unsafe_allow_html=True)
description = st.text_area("", placeholder="Enter keywords or descriptions of temples you seek...", label_visibility="collapsed")

# Buttons in columns
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Discover Temples"):
        recommendations = get_recommendations(description)
        st.session_state.show_recommendations = True
        st.session_state.recommendations = recommendations

with col2:
    if st.button("Show All Locations"):
        st.session_state.show_map = True

with col3:
    if st.button("Sacred Heritage Sites"):
        st.session_state.show_top_places = True

# Display recommendations
if st.session_state.show_recommendations and 'recommendations' in st.session_state:
    recommendations = st.session_state.recommendations
    
    if recommendations.shape[0] > 0:
        st.markdown('<h2 class="section-header">RECOMMENDED SACRED SITES</h2>', unsafe_allow_html=True)
        
        for _, row in recommendations.iterrows():
            st.markdown(f"""
            <div class="temple-card">
                <div class="temple-name">{row['templeName']}</div>
                <div class="temple-description">{row['Description']}</div>
                <div class="temple-coordinates">Coordinates: {row['Coordinates']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<p class="temple-description" style="text-align: center;">No temples found matching your description. Please refine your search.</p>', unsafe_allow_html=True)

# Display the map
if st.session_state.show_map:
    st.markdown('<h2 class="section-header">HERITAGE TEMPLES OF INDIA</h2>', unsafe_allow_html=True)
    
    # Convert Coordinates to numeric
    coordinates = pd.DataFrame(
        df['Coordinates'].apply(parse_coordinates).tolist(),
        columns=['Latitude', 'Longitude']
    )
    df[['Latitude', 'Longitude']] = coordinates
    df_map = df.dropna(subset=['Latitude', 'Longitude'])
    
    # Create a folium map
    m = folium.Map(
        location=[df_map['Latitude'].mean(), df_map['Longitude'].mean()], 
        zoom_start=5,
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        min_zoom=4,
        max_bounds=True,
        world_copy_jump=False
    )
    
    # Add markers
    for _, row in df_map.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(f"<b style='color: #d4af37; font-family: Cinzel;'>{row['templeName']}</b>", max_width=300),
            icon=folium.Icon(color='orange', icon='place', prefix='fa')
        ).add_to(m)
    
    st_folium(m, width=1200, height=600, returned_objects=[])

# Display top places
if st.session_state.show_top_places:
    st.markdown('<h2 class="section-header">REVERED MONUMENTS OF INDIA</h2>', unsafe_allow_html=True)
    
    for _, row in static_top_places.iterrows():
        st.markdown(f"""
        <div class="temple-card">
            <div class="temple-name">{row['TempleName']}</div>
            <div class="temple-description">{row['Description']}</div>
            <div class="temple-coordinates">Coordinates: {row['Coordinates']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Convert Coordinates to numeric
    coordinates = pd.DataFrame(
        static_top_places['Coordinates'].apply(parse_coordinates).tolist(),
        columns=['Latitude', 'Longitude']
    )
    static_top_places[['Latitude', 'Longitude']] = coordinates
    top_places_map = static_top_places.dropna(subset=['Latitude', 'Longitude'])
    
    # Create a folium map
    m = folium.Map(
        location=[top_places_map['Latitude'].mean(), top_places_map['Longitude'].mean()], 
        zoom_start=5,
        tiles='https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
        attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        min_zoom=4,
        max_bounds=True,
        world_copy_jump=False
    )
    
    # Add markers
    for _, row in top_places_map.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=folium.Popup(f"<b style='color: #d4af37; font-family: Cinzel;'>{row['TempleName']}</b>", max_width=300),
            icon=folium.Icon(color='orange', icon='place', prefix='fa')
        ).add_to(m)
    
    st_folium(m, width=1200, height=600, returned_objects=[])
