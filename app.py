import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import streamlit.components.v1 as components

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn

def plot_hexbin(x, y, data=None, cmap="YlGnBu_r", bins=30, figsize=(10, 8), title=None):
    """
    Concise Hexbin Plot Drawing Function
    
    Parameters:
    x, y: Data column names or arrays
    data: DataFrame (optional)
    cmap: Color mapping
    bins: Number of hexagons
    figsize: Figure size
    title: Plot title
    """
    # Process Data
    if data is not None:
        x_data = data[x]
        y_data = data[y]
        x_label = x
        y_label = y
    else:
        x_data = np.array(x)
        y_data = np.array(y)
        x_label = "x"
        y_label = "y"
        data = pd.DataFrame({x_label: x_data, y_label: y_data})
    
    #  Create figure
    plt.figure(figsize=figsize, dpi=100)
    
    # Calculate aspect ratio
    width, height = figsize
    ratio = int(height * 0.8)
    
    # Draw hexbin plot
    g = sns.jointplot(
        data=data, 
        x=x_label, 
        y=y_label,
        kind='hex',
        height=height * 0.9,
        ratio=ratio,
        space=0.1,
        color="#2ca02c",
        gridsize=bins,
        cmap=cmap
    )
    
    #Set title
    if title:
        g.fig.suptitle(title, y=1.02)
    
    # Optimize style
    g.fig.set_facecolor('white')
    g.ax_joint.set_facecolor('white')
    
    # Calculate correlation coefficient
    corr = np.corrcoef(x_data, y_data)[0, 1]
    g.ax_joint.text(0.05, 0.95, f'r = {corr:.3f}', 
                    transform=g.ax_joint.transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return g.fig

# --- Page Configuration ---
st.set_page_config(page_title="Music Popularity In-depth Analysis System", layout="wide", page_icon="🎵")

# --- 1. Inject CustCSS ---
st.markdown("""
    <style>
    /* Basic Layout Optimization */
    .stMainBlockContainer { margin-top: -70px; }
    
    /* Centered Main Title */
    .main-header {
        text-align: center;
        color: #1DB954;
        font-weight: 800;
        font-size: 1.5rem;
        margin-bottom: 20px;
    }
    
    /* Analysis Report Text Card  */
    .report-card {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #e1e4e8;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        color: #2D3436;
        line-height: 1.8;
    }
    
    /* Highlight Prediction Result */
    .prediction-box {
        background: #f1f2f6;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #1DB954;
    }
    </style>
    """, unsafe_allow_html=True)

# Fixed Top Navigation Bar
components.html("""
<div style="height: 60px; background: #1DB954; line-height: 60px; text-align: center; 
            font-size: 26px; color: white; font-weight: bold; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            position: fixed; top: 0; left: 0; width: 100%; z-index: 9999;">
    🎵 Music Popularity In-depth Analysis System</div>
""", height=0)

st.sidebar.html("<div style='text-align: center; font-weight: bold; font-size: 1.5rem;'>🎵 Music Popularity In-depth Analysis System</div><hr>")

# --- 2. Load Resources ---
@st.cache_resource
def load_assets():
   # Get the folder path where app.py is located (Core: works no matter where the teacher runs it)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Splice into complete path (Key: variables here will be used later!)
    model_path    = os.path.join(base_dir, "music_popularity_rf(1).pkl")
    feature_path  = os.path.join(base_dir, "feature_columns(1).pkl")
    test_path     = os.path.join(base_dir, "test.pkl")
    csv_path      = os.path.join(base_dir, "Spotify-2000.csv")

    # When loading files, use the spliced path variables above!!!
    model = joblib.load(model_path)
    features = joblib.load(feature_path)
    test = joblib.load(test_path)
    df = pd.read_csv(csv_path)

    explainer = shap.TreeExplainer(model)
    analyzer = SentimentIntensityAnalyzer()

    return model, features, explainer, analyzer, test, df

# # Add exception handling during loading for user-friendly prompts for the instructor
try:
    model, feature_columns, explainer, analyzer, test, df = load_assets()
except FileNotFoundError as e:
    st.error(f"❌ Error: File not found! Please ensure all files are in the same folder as app.py. \nDetailed error: {e}")
    st.stop()
    st.stop()
except Exception as e:
    st.error(f"❌ Failed to load resources: {e}")
    st.stop()    

df["Length (Duration)"] = df["Length (Duration)"].apply(lambda x: int(str(x).replace(",","")))

# Define high popularity threshold (e.g., top 25%)
high_pop_threshold = df['Popularity'].quantile(0.75)

# Get feature statistics of high popularity songs
high_pop_songs = df[df['Popularity'] >= high_pop_threshold]

# Calculate feature ranges for high popularity song
feature_ranges = {}
for feature in ['Beats Per Minute (BPM)', 'Energy', 'Danceability', 
                'Loudness (dB)', 'Liveness', 'Valence', 
                'Length (Duration)', 'Acousticness', 'Speechiness']:
    feature_ranges[feature] = {
        'min': high_pop_songs[feature].min(),
        'max': high_pop_songs[feature].max(),
        'mean': high_pop_songs[feature].mean(),
        'std': high_pop_songs[feature].std(),
        'percentile_25': high_pop_songs[feature].quantile(0.25),
        'percentile_75': high_pop_songs[feature].quantile(0.75)
    }

# Title Length and Sentiment Analysis
high_pop_songs['Title Length'] = high_pop_songs['Title'].str.len()
title_length_stats = {
    'min': high_pop_songs['Title Length'].min(),
    'max': high_pop_songs['Title Length'].max(),
    'mean': high_pop_songs['Title Length'].mean(),
    'percentile_25': high_pop_songs['Title Length'].quantile(0.25),
    'percentile_75': high_pop_songs['Title Length'].quantile(0.75)
}
    
# --- 3. Sidebar Button Interaction---
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.image("https://img.shetu66.com/2023/07/11/1689059685175876.png", use_container_width=True)
st.sidebar.markdown("### 🛠️ 🎛️ System Control Panel")

if "menu" not in st.session_state:
    st.session_state["menu"] = "Metrics"

if st.sidebar.button("📊 Data Model Performance Evaluation", use_container_width=True):
    st.session_state["menu"] = "Metrics"
if st.sidebar.button("🔍 Feature Driver Factor Analysis", use_container_width=True):
    st.session_state["menu"] = "Importance"
if st.sidebar.button("🔮 Popularity Simulation Prediction", use_container_width=True):
    st.session_state["menu"] = "Prediction"
if st.sidebar.button("💻 Feature Distribution Exploration Analysis", use_container_width=True):
    st.session_state["menu"] = "cumputer"

# --- 4. Module Logic ---

# Module 1: Model Performance
if st.session_state["menu"] == "Metrics":
    st.markdown("<div class='main-header'>📊 Model Performance and Prediction Benchmark Evaluation</div>", unsafe_allow_html=True)
    st.html("<hr>")
    col1, col2, col3 = st.columns(3, border=True)
    col1.metric("R² Score (Explanatory Power)", "0.325", "Pass Level")
    col2.metric("MAE (Mean Absolute Error)", "9.74", "-10%Deviation")
    col3.metric("RMSE (Root Mean Squared Error)", "12.04", "Large Deviation")
    st.html("<hr>")
    with st.container(border=True):
        left, right = st.columns([3, 2], gap="large")

    with left:
        st.subheader("🎯 Actual vs. Predicted Value Distribution Chart")

        fig = px.scatter(x=test[0], y=test[1], labels={'x':'Actual Popularity', 'y':'Predicted Popularity'},
                         template="plotly_white", opacity=0.7)
        fig.add_shape(type="line", x0=20, y0=20, x1=95, y1=95, line=dict(color="Red", dash="dash"))
        fig.update_traces(marker=dict(size=10, color='#1DB954'))
        
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("📝 In-depth Analysis Conclusion")
        st.markdown(f"""
        <div>
            <b>1. Overall Explanatory Power Evaluation</b><br>
            The R² of the random forest regression model is 0.325, indicating that the model has captured 32.5% of the fluctuation logic in the dataset. It has initially established a prediction benchmark for popularity trends despite the highly complex and subjective market context.<br><br>
            <b>2. Error Trend Interpretation</b><br>
            MAE (9.74) shows good performance, but RMSE (12.04) is relatively high, revealing a "black swan effect" in the dataset — that is, some songs, although average in inherent features, achieved extremely high popularity due to external factors, leading to biased predictions.
        </div>
        """, unsafe_allow_html=True)

# Module 2: Feature Analysis
elif st.session_state["menu"] == "Importance":
    st.markdown("<div class='main-header'>🔍 Popularity Driver Factor Ladder Analysis</div>", unsafe_allow_html=True)
    st.html("<hr>")
    imp_data = {
        "Feature": ["Year", "Length", "Dutch Indie", "Danceability", "Liveness", "Loudness", "BPM", "Valence", "Title Length", "Energy"],
        "Importance": [0.118, 0.090, 0.084, 0.075, 0.074, 0.074, 0.070, 0.066, 0.065, 0.061]
    }
    df_plot = pd.DataFrame(imp_data).sort_values("Importance", ascending=True)
    
    fig_imp = px.bar(df_plot, x='Importance', y='Feature', orientation='h', 
                     color='Importance', color_continuous_scale='Greens',
                     template="plotly_white", title="Global Influence Weight of Core Features")
    
    with st.container(border=True):
        st.plotly_chart(fig_imp, use_container_width=True)


    t1, t2, t3 = st.columns(3, border=True)
    t1.write("🕒 **First Tier: Era Characteristics**\nThe influence of release year ranks first (0.118), proving that the popularity of music has a strong periodicity, and the phenomenon of classics and new hits is reflected in the model.")
    t2.write("💃 **Second Tier: Rhythm Dynamics**\nDanceability and Loudness have extremely close weights, forming the core physical indicators for modern commercial pop music to capture the audience's ears in the first place.")
    t3.write("🏷️ **Third Tier: Market Labels**\nDutch Indie shows extremely high market certainty, indicating that in niche vertical fields, the predictive benefits brought by genre labels even exceed basic energy features (Energy).")

# Module 3: Simulation Prediction
elif st.session_state["menu"] == "Prediction":
    st.markdown("<div class='main-header'>🎵 Single Song Popularity AI Prediction Tool</div>", unsafe_allow_html=True)
    st.html("<hr>")
    
    with st.container():
        with st.form("predict_form"):
            st.markdown("##### 🎹 Please configure attribute parameters for the simulated song")
            c1, c2, c3 = st.columns(3)
            with c1:
                title = st.text_input("Song Name", "Top Hit Song")
                year = st.number_input("Expected Release Year", 1950, 2026, 2024)
                bpm = st.slider("BPM (Beats Per Minute)", 50, 200, 118)
            with c2:
                duration = st.number_input("Duration (seconds)", 60, 600, 205)
                dance = st.slider("Danceability ", 0.0, 1.0, 0.65)
                energy = st.slider("Energy", 0.0, 1.0, 0.72)
            with c3:
                genre = st.selectbox("Core Genre Category", ["dutch indie", "dutch pop", "album rock", "other"])
                loudness = st.slider("Loudness (dB)", -60, 0, -7)
                valence = st.slider("Valence (Positivity)", 0.0, 1.0, 0.55)
            
            # Add More Feature Inputs
            st.markdown("##### 🎼 More Music Feature Settings")
            c4, c5, c6 = st.columns(3)
            liveness = c4.slider("Liveness", 0, 100, 20)
            acousticness = c5.slider("Acousticness", 0, 100, 30)
            speechiness = c6.slider("Speechiness ", 0, 100, 5)
            
            submit = st.form_submit_button("🚀 Run AI Popularity Prediction", use_container_width=True)

    if submit:
        # Feature Engineering Processing
        input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
        input_df['Year'] = year
        input_df['Length (Duration)'] = duration
        input_df['Beats Per Minute (BPM)'] = bpm
        input_df['Danceability'] = int(dance * 100)  # Convert to 0-100
        input_df['Energy'] = int(energy * 100)  # Convert to 0-100
        input_df['Loudness (dB)'] = loudness
        input_df['Valence'] = int(valence * 100)  # Convert to 0-100
        input_df['Liveness'] = liveness
        input_df['Acousticness'] = acousticness
        input_df['Speechiness'] = speechiness
        input_df['Title Length'] = len(title)
        input_df['Title Emotion'] = analyzer.polarity_scores(title)["compound"]
        
        genre_col = f"Genre_Grouped_{genre}"
        if genre_col in input_df.columns:
            input_df[genre_col] = 1

        # Run Model Prediction
        prediction = model.predict(input_df)[0]
        shap_values = explainer.shap_values(input_df)
        
        st.html("<hr>")
        
        # Display Prediction Result
        with st.container(border=True):
            st.markdown("#### 📊 Simulation Prediction Result Analysis")
            
            if prediction >= 70:
                st.success(f"🎯 Predicted Popularity Score: {prediction:.2f} | 🌟 Has Hit Potential")
            elif prediction >= 60:
                st.info(f"🎯 Predicted Popularity Score:  {prediction:.2f} | ⭐ Above Average Level")
            elif prediction >= 50:
                st.warning(f"🎯 Predicted Popularity Score:  {prediction:.2f} | 🎵 Niche Track")
            else:
                st.error(f"🎯 Predicted Popularity Score: {prediction:.2f} | ⚠️ Needs Optimization")
            
            st.warning("**Note: Model results are for reference only. Suggestions have no direct correlation with the model. It is recommended to analyze rationally by combining both.**")
            
        # Feature Range Check & Suggestion
        with st.container(border=True):
            st.markdown("#### 🔍 Feature Optimization Suggestion")
            
            suggestions = []
            
            # Check if Music Features Are Within the Range of Popular Songs
        features_to_check = {
        'Beats Per Minute (BPM)': bpm,
        'Energy': int(energy * 100),
        'Danceability': int(dance * 100),
        'Loudness (dB)': loudness,
        'Liveness': liveness,
        'Valence': int(valence * 100),
        'Length (Duration)': duration,
        'Acousticness': acousticness,
        'Speechiness': speechiness
}

        suggestions = []            
        for feature, value in features_to_check.items():
            if feature in feature_ranges:
                range_info = feature_ranges[feature]

        # Check if within common range (25%-75% percentile)
                if value < range_info['percentile_25']:
                    suggestions.append(f"📉 **{feature}**: {value} (Low, popular songs usually between {int(range_info['percentile_25'])}-{int(range_info['percentile_75'])})")
                elif value > range_info['percentile_75']:
                    suggestions.append(f"📈 **{feature}**: {value} (High, popular songs usually between {int(range_info['percentile_25'])}-{int(range_info['percentile_75'])})")            
            # Check Title Features
        title_len = len(title)
        if title_len < title_length_stats['percentile_25']:
            suggestions.append(f"📈 **Title Length**: {title_len}chars (Short, recommend {int(title_length_stats['percentile_25'])}-{int(title_length_stats['percentile_75'])}chars)")
        elif title_len > title_length_stats['percentile_75']:
            suggestions.append(f"📉 **Title Length**: {title_len}chars (Long, recommend {int(title_length_stats['percentile_25'])}-{int(title_length_stats['percentile_75'])}chars)")
            
            #  Title Sentiment Analysis
        title_emotion = analyzer.polarity_scores(title)["compound"]
        if title_emotion < -0.1:
            suggestions.append(f"😔 **Title Sentiment**: Negative (Recommend more positive/neutral words)")
        elif title_emotion < 0.1:
            suggestions.append(f"😐 **Title Sentiment**: Neutral (Neutral titles are usually safer)")
        else:
            suggestions.append(f"😊 **Title Sentiment**: Positive (Positive titles are more attractive)")
            
            # Display Suggestions
        if suggestions:
            st.markdown("##### 💡 Optimization Suggestion:")
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")
        else:
            st.success("✅ Your song features are reasonably set, consistent with the feature range of mainstream popular songs!")
                
            # Display Feature Statistics
        with st.expander("📈 Popular Song Feature Statistics Reference:"):
            col1, col2, col3 = st.columns(3)
            features_display = list(features_to_check.keys())[:9]
            for i, feature in enumerate(features_display):
                if feature in feature_ranges:
                    with [col1, col2, col3][i % 3]:
                        range_info = feature_ranges[feature]
                        st.metric(
                            label=f"{feature}",
                            value=f"{features_to_check[feature]}",
                            delta=f"Ref: {int(range_info['percentile_25'])}-{int(range_info['percentile_75'])}"
                            )
        
        # SHAP Attribution Analysis
        with st.container(border=True):
            st.markdown("#### 🎯 Feature Attribution Path (SHAP Force Analysis)")
            st.markdown("##### Red represents factors boosting popularity, Blue represents factors reducing popularity.")
            
            # Render SHAP Force Plot
            fig_shap, ax = plt.subplots(figsize=(12, 3), dpi=100)
            shap.force_plot(explainer.expected_value, shap_values[0], input_df.iloc[0], 
                            matplotlib=True, show=False, text_rotation=0)
            st.pyplot(plt.gcf(), use_container_width=True)
            
            # Add Feature Importance Interpretation
            st.markdown("##### 📊 Feature Impact Analysis:")
            
            # Get features with the largest impact on prediction
            shap_df = pd.DataFrame({
                'feature': input_df.columns,
                'shap_value': shap_values[0]
            })
            shap_df['abs_impact'] = shap_df['shap_value'].abs()
            top_features = shap_df.nlargest(5, 'abs_impact')
            
            cols = st.columns(5)
            for idx, (_, row) in enumerate(top_features.iterrows()):
                with cols[idx]:
                    impact_type = "Boost" if row['shap_value'] > 0 else "Reduce"
                    color = "green" if row['shap_value'] > 0 else "red"
                    st.markdown(f"<div style='text-align: center;'>"
                               f"<div style='color: {color}; font-size: 20px;'>{'↑' if row['shap_value'] > 0 else '↓'}</div>"
                               f"<div><strong>{row['feature'][:15]}</strong></div>"
                               f"<div style='color: {color};'>{impact_type}popularity</div>"
                               f"</div>", unsafe_allow_html=True)
        
        # Plot Radar Chart with Plotly
        with st.container(border=True):
            st.markdown("#### 📈 Song Feature Comparison Radar Chart")
            
            #  Select Features for Comparison
            features_for_radar = ['Energy', 'Danceability', 'Valence', 
                                 'Loudness (dB)', 'Beats Per Minute (BPM)', 'Acousticness']
            
            # Prepare Data
            user_values = []
            high_pop_values = []
            feature_names_clean = []
            
            for feature in features_for_radar:
                if feature in features_to_check:
                    user_values.append(features_to_check[feature])
                if feature in feature_ranges:
                    high_pop_values.append(feature_ranges[feature]['mean'])
                feature_names_clean.append(feature.replace(' (', '\n('))
            
            # Normalize Each Feature
            user_norm = []
            high_pop_norm = []
            
            for i, feature in enumerate(features_for_radar):
                if feature in feature_ranges:
                    # Get Feature Range
                    if feature in ['Loudness (dB)']:
                        # Special Handling for Loudness Range
                        max_val = 0
                        min_val = -30
                    elif feature in ['Beats Per Minute (BPM)']:
                        max_val = 200
                        min_val = 50
                    else:
                        max_val = 100
                        min_val = 0
                    
                    # Normalize to 0-1 range
                    user_norm.append((user_values[i] - min_val) / (max_val - min_val))
                    high_pop_norm.append((high_pop_values[i] - min_val) / (max_val - min_val))
            
            # Close the graph
            user_norm_closed = user_norm + [user_norm[0]]
            high_pop_norm_closed = high_pop_norm + [high_pop_norm[0]]
            feature_names_closed = feature_names_clean + [feature_names_clean[0]]
            
            # Create Plotly Radar Chart
            fig = go.Figure()
            
            # Add User Song Data
            fig.add_trace(go.Scatterpolar(
                r=user_norm_closed,
                theta=feature_names_closed,
                name='Your Song',
                fill='toself',
                fillcolor='rgba(255, 107, 107, 0.3)',
                line=dict(color='rgb(255, 107, 107)', width=3),
                marker=dict(size=6, color='rgb(255, 107, 107)'),
                hovertemplate='<b>%{theta}</b><br>Your Value: %{customdata[0]:.1f}<br>Normalized:%{r:.3f}<extra></extra>',
                customdata=np.column_stack([user_values + [user_values[0]], user_norm_closed])
            ))
            
            # Add High Popularity Song Average Dat
            fig.add_trace(go.Scatterpolar(
                r=high_pop_norm_closed,
                theta=feature_names_closed,
                name='High Popularity Avg:',
                fill='toself',
                fillcolor='rgba(30, 136, 229, 0.2)',
                line=dict(color='rgb(30, 136, 229)', width=3, dash='dash'),
                marker=dict(size=6, color='rgb(30, 136, 229)'),
                hovertemplate='<b>%{theta}</b><br>High Popularity Avg: %{customdata[0]:.1f}<br>Normalized: %{r:.3f}<extra></extra>',
                customdata=np.column_stack([high_pop_values + [high_pop_values[0]], high_pop_norm_closed])
            ))
            
            # Update Chart Layout
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickvals=[0, 0.25, 0.5, 0.75, 1],
                        ticktext=['Low', 'Lower', 'Medium', 'Higher', 'High'],
                        gridcolor='rgba(200, 200, 200, 0.5)',
                        linecolor='rgba(200, 200, 200, 0.8)',
                        showline=True,
                        linewidth=1
                    ),
                    angularaxis=dict(
                        gridcolor='rgba(200, 200, 200, 0.5)',
                        linecolor='rgba(200, 200, 200, 0.8)',
                        rotation=90
                    ),
                    bgcolor='rgba(250, 250, 250, 0.5)'
                ),
                title=dict(
                    text='Song Feature Comparison Radar Chart',
                    x=0.5,
                    y=0.95,
                    xanchor='center',
                    yanchor='top',
                    font=dict(size=20, color='#333333', family="Arial")
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.07,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12, family="Arial"),
                    bgcolor='rgba(255, 255, 255, 0.8)',
                    bordercolor='rgba(200, 200, 200, 0.8)',
                    borderwidth=1
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=500,
                margin=dict(t=100, b=50, l=50, r=50),
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': True})
            
            # Add Feature Comparison Table
            with st.expander("📊 View Detailed Feature Comparison"):
                st.markdown("**Feature Value Comparison Table**")
                
                comparison_data = []
                for i, feature in enumerate(features_for_radar):
                    if feature in features_to_check and feature in feature_ranges:
                        user_val = user_values[i]
                        high_pop_val = int(high_pop_values[i])
                        diff = user_val - high_pop_val
                        diff_percent = (diff / high_pop_val * 100) if high_pop_val != 0 else 0
                        
                        status_icon = "🟢" if abs(diff_percent) < 10 else "🟡" if abs(diff_percent) < 20 else "🔴"
                        status_text = "Close" if abs(diff_percent) < 10 else "Difference" if abs(diff_percent) < 20 else "Large Difference"
                        
                        comparison_data.append({
                            "Feature": feature,
                            "Your Value": user_val,
                            "High Pop Avg": high_pop_val,
                            "Difference": f"{diff:+d}",
                            "Diff %": f"{diff_percent:+.1f}%",
                            "Status": f"{status_icon} {status_text}"
                        })
                
                # Create Comparison Table
                comparison_df = pd.DataFrame(comparison_data)
                
                # Use Plotly Table
                fig_table = go.Figure(data=[go.Table(
                    columnwidth=[120, 80, 100, 60, 100, 100],
                    header=dict(
                        values=list(comparison_df.columns),
                        fill_color='rgb(30, 136, 229)',
                        align='center',
                        font=dict(color='white', size=12, family="Arial"),
                        height=40
                    ),
                    cells=dict(
                        values=[comparison_df[col] for col in comparison_df.columns],
                        fill_color=['rgba(245, 245, 245, 0.8)', 'rgba(255, 255, 255, 0.8)'],
                        align=['left', 'center', 'center', 'center', 'center', 'center'],
                        font=dict(color='#333333', size=11, family="Arial"),
                        height=30
                    )
                )])
                
                fig_table.update_layout(
                    height=250,
                    margin=dict(t=0, b=0, l=0, r=0)
                )
                
                st.plotly_chart(fig_table, use_container_width=True)
                
                # Add Explanation
                st.markdown("""
                **📊 Chart Legend Explanation:**
                - **Solid filled area**: Your song features
                - **Dashed filled area**: Average features of high-popularity songs
                - **Far from center**: Indicates higher feature value
                - **Color meaning**: Red area represents your song, Blue area represents high-popularity average                
                - **Status explanation**:🟢 Small difference(<10%), 🟡 Different(10-20%), 🔴 Large difference(>20%)                """)
                
# # Module 4: Feature Distribution Exploration Analysis
if st.session_state["menu"] == "cumputer":
    st.markdown("<div class='main-header'>💻 Feature Distribution Exploration Analysis</div>", unsafe_allow_html=True)
    st.html("<hr>")
    
    df["Length (Duration)"] = df["Length (Duration)"].apply(lambda x: int(str(x).replace(",","")))

    numeric_col = "Year	Beats Per Minute (BPM)	Energy	Danceability	Loudness (dB)	Liveness	Valence	Length (Duration)	Acousticness	Speechiness	Popularity".split("\t")

    col = st.columns(2)
    x1 = col[0].selectbox("X-axis", numeric_col)
    y1 = col[1].selectbox("Y-axis", numeric_col)
    
    fig = plot_hexbin(x1, y1, data=df, cmap="Reds", bins=30, figsize=(10, 8), title=f"{x1} vs {y1}")
    
    col = st.columns([1,3,1])
    col[1].pyplot(fig, use_container_width=True)
    
with st.sidebar.container(border=True):
    st.html("""
    <div style='text-align: center;color:red;'>System Description</div><hr>
    <p style="font-size: 10px;">This system aims to analyze features affecting music popularity through dataset exploration. The core objective is to interpret music popularity through model feature importance.   However, factors affecting music popularity include but are not limited to dataset features. Many social factors lead to significant prediction errors.   Nevertheless, feature importance has good indicative value for music popularity, which can provide an evaluative reference system for music trend prediction.</p>
    <div style="text-align:center; color:red; font-size:10px;">Results are for reference only; not applicable for major decision-making!!!</div>
    """)
