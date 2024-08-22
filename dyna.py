import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import folium
from streamlit_folium import st_folium


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
        desc_scores = [score for score in desc_scores if score[1] > 0.0]  # Filter scores > 0.0
        desc_scores = sorted(desc_scores, key=lambda x: x[1], reverse=True)
        desc_scores = desc_scores[:10]  # Top 10 similar based on description
        desc_indices = [i[0] for i in desc_scores]
        recommendations = df.iloc[desc_indices][[
            'templeName', 'Coordinates', 'Description'
        ]]

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

# Streamlit UI
st.title("Temple Recommendation System")

# Apply custom CSS for background gradient
st.markdown("""
    <style>
    .folium-map {
        width: 200px;
        height: 100px;
        margin: auto;
        padding: 0;
    }
    .streamlit-expander {
        padding: 0;
    }

    .stApp {
        background: linear-gradient(to right, #ff9a9e, #fad0c4); /* Gradient pink and light blue */
    }
    .stTitle {
        color: #333; /* Title color */
        font-size: 2em; /* Title size */
    }
    .stButton > button {
        background-color: #007bff; /* Button background color */
        color: #fff; /* Button text color */
        border: none;
        padding: 10px 20px;
        border-radius: 5px;
        cursor: pointer;
    }
    .stButton > button:hover {
        background-color: #0056b3; /* Button hover color */
    }
    </style>
""", unsafe_allow_html=True)



# User input for description
description = st.text_area("Enter Temple Description:")

# Recommendation button
if st.button("Recommend"):
    recommendations = get_recommendations(description)
    st.session_state.show_recommendations = True
    st.session_state.recommendations = recommendations

# Display recommendations if button pressed
if st.session_state.show_recommendations and 'recommendations' in st.session_state:
    recommendations = st.session_state.recommendations
    if recommendations.shape[0] > 0:
        st.write("### Recommended Temples")
        st.write("Here are some temples you might find interesting based on your description:")

        # Format recommendations in a styled way
        for _, row in recommendations.iterrows():
            st.write(f"🌟 **Temple Name**: {row['templeName']}")
            st.write(f"🗺️ **Description**: {row['Description']}")
            st.write(f"📍 **Location Coordinates**: {row['Coordinates']}")
            st.write("---")
    else:
        st.write("No recommendations found. Please try a different description.")

# Show All Locations button
if st.button("Show All Locations"):
    st.session_state.show_map = True

# Display the map if button pressed
if st.session_state.show_map:
    st.write("### All Temples in the Dataset")
    st.write("Here are all the temples with their locations:")

    # Convert Coordinates to numeric for plotting on the map
    coordinates = pd.DataFrame(
        df['Coordinates'].apply(parse_coordinates).tolist(),
        columns=['Latitude', 'Longitude']
    )
    df[['Latitude', 'Longitude']] = coordinates

    # Remove rows with NaN coordinates
    df = df.dropna(subset=['Latitude', 'Longitude'])

    # Create a folium map
    m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=6)

    # Add markers with temple names
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=row['templeName'],
            icon=folium.Icon(icon='info-sign')
        ).add_to(m)

    # Display the map in Streamlit
    st_folium(m, width=700, height=500)

# Top Places button
if st.button("Top Places"):
    st.session_state.show_top_places = True

# Display top places if button pressed
if st.session_state.show_top_places:
    st.write("### Top Temples and Monuments in India")
    st.write("Here are some top temples and monuments in India:")

    # Display static top places
    for _, row in static_top_places.iterrows():
        st.write(f"🌟 **Temple Name**: {row['TempleName']}")
        st.write(f"🗺️ **Description**: {row['Description']}")
        st.write(f"📍 **Location Coordinates**: {row['Coordinates']}")
        st.write("---")

    # Convert Coordinates to numeric for plotting on the map
    coordinates = pd.DataFrame(
        static_top_places['Coordinates'].apply(parse_coordinates).tolist(),
        columns=['Latitude', 'Longitude']
    )
    static_top_places[['Latitude', 'Longitude']] = coordinates

    # Remove rows with NaN coordinates
    static_top_places = static_top_places.dropna(subset=['Latitude', 'Longitude'])

    # Create a folium map
    m = folium.Map(location=[static_top_places['Latitude'].mean(), static_top_places['Longitude'].mean()], zoom_start=6)

    # Add markers with temple names
    for _, row in static_top_places.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=row['TempleName'],
            icon=folium.Icon(icon='info-sign')
        ).add_to(m)

    # Display the map in Streamlit
    st_folium(m, width=200, height=100)
