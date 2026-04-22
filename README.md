# songs-popularity-predictor
Music Popularity Prediction Assistant


A data-driven interactive tool that predicts song popularity scores based on audio features and provides actionable optimization recommendations for music creators.


Project Overview


This project addresses a critical challenge in the music industry: record labels and independent artists invest substantial resources in music production yet lack data-driven indicators of popularity potential. The solution helps music creators make informed decisions during the production phase by predicting popularity scores and identifying optimization opportunities before release.
Target Users


Independent music producers seeking data-driven guidance for creative direction
A&R departments responsible for talent scouting and signing decisions
Music marketing teams developing promotional strategies
Core Features


Real-time popularity prediction based on audio features
SHAP-based feature attribution explaining prediction logic
Radar chart comparison against high-popularity benchmarks
Numerical feature comparison tables with gap analysis
Tailored optimization recommendations based on popular song statistics


Files Description
Application Files


File	Description
app.py	Main Streamlit application with 4 interactive modules
music_popularity_rf.pkl	Pre-trained Random Forest regression model
feature_columns.pkl	Feature list used for prediction
test.pkl	Test data containing actual and predicted values
Data File (Required to Download)


File	Description	Source
Spotify-2000.csv	Spotify Top 2000s Mega Dataset with 1,994 songs and 15 features	Download from Kaggle
Supporting Files


File	Description
requirements.txt	Python dependencies list
README.md	This documentation file
Reflection_Report.pdf	Project reflection report


Project Structure


plaintext
music-popularity-prediction/
│
├── app.py                         # Main application
├── music_popularity_rf.pkl        # Trained model
├── feature_columns.pkl            # Feature list
├── test.pkl                       # Test data
├── Spotify-2000.csv               # Dataset (download from Kaggle)
│
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
└── Reflection_Report.pdf          # Reflection report



Note: All files must be placed in the same directory. The application automatically locates files relative to app.py.


Requirements
Python Version


Python 3.8 or higher
Dependencies


txt
streamlit>=1.0.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
joblib>=1.0.0
shap>=0.40.0
nltk>=3.6.0
plotly>=5.0.0
seaborn>=0.11.0
matplotlib>=3.4.0



Installation & Setup
Step 1: Clone Repository


bash
git clone [repository-url]
cd music-popularity-prediction

Step 2: Download Dataset


Visit Kaggle - Spotify Top 2000s Mega Dataset
Download the dataset CSV file
Rename it to Spotify-2000.csv
Place it in the same directory as app.py
Step 3: Install Dependencies


bash
pip install -r requirements.txt



Or install manually:


bash
pip install streamlit pandas numpy scikit-learn joblib shap nltk plotly seaborn matplotlib



Running the Application
Quick Start


bash
# Navigate to the project directory (where app.py is located)
cd [project-directory]

# Run Streamlit application
streamlit run app.py

Access the Application


After running the command, open your browser and visit:


plaintext
http://localhost:8501



Application Modules


The Streamlit application contains four interactive modules:
1. Data Model Performance Evaluation 📊


Purpose: Evaluate model reliability and prediction accuracy


Features:


Core Metrics Display: R² Score (0.3025), MAE (9.8946), RMSE (12.2407)
Actual vs. Predicted Scatter Plot: Visualize prediction distribution with ideal fit line
In-depth Analysis Conclusion:
Overall explanatory power evaluation
Error trend interpretation with "black swan effect" analysis


Key Insights:


R²=0.3025 indicates the model captures 30.25% of popularity variance
High RMSE reveals outliers from external factors (marketing, social media virality)


2. Feature Driver Factor Analysis 🔍


Purpose: Understand which features drive music popularity


Features:


Global Influence Weight Bar Chart: Top 10 features ranked by importance
Three-Tier Feature Framework:
First Tier (Era Characteristics): Year (0.118) - proves music popularity has strong periodicity
Second Tier (Rhythm Dynamics): Danceability & Loudness - core physical indicators for commercial pop
Third Tier (Market Labels): Dutch Indie genre shows high market certainty in niche verticals


Key Insights:


Release year is the most influential predictor
Danceability and Loudness are equally important for modern pop music
Genre labels can outperform basic audio features in prediction value


3. Popularity Simulation Prediction 🔮 (Main Feature)


Purpose: Predict popularity for custom songs and provide optimization guidance


Step 1: Configure Song Parameters


Users can adjust 12+ features via interactive sliders and inputs:


Basic Info: Song Name, Release Year
Core Audio Features: BPM, Danceability, Energy, Loudness, Valence
Advanced Features: Liveness, Acousticness, Speechiness
Genre Selection: Dutch Indie, Dutch Pop, Album Rock, Other


Step 2: View Prediction Result


Popularity Score: Instant prediction with rating (Hit Potential / Above Average / Niche Track / Needs Optimization)
Confidence Indicator: Clear status labels for decision-making


Step 3: Feature Optimization Suggestions


The system analyzes user's song against high popularity song statistics (top 25th percentile):


Detects features outside optimal range (25th-75th percentile of popular songs)
Provides specific numerical references (e.g., "Liveness: 20 (High, popular songs usually between 9-19)")
Analyzes title sentiment using NLTK VADER (Positive/Neutral/Negative)


Step 4: SHAP Force Analysis 🎯


Revolutionary feature: Explains WHY the model made this prediction


Red bars: Features boosting popularity
Blue bars: Features reducing popularity
Base value to prediction: Visual path showing how features shift the prediction
Top 5 Impact Features: Clear arrows indicating boost/reduce direction


Step 5: Radar Chart Comparison 📈


Visual comparison between your song and high popularity average:


6 dimensions: Energy, Danceability, Valence, Loudness, BPM, Acousticness
Interactive hover: Shows exact values and normalized scores
Intuitive visualization: Red area (your song) vs Blue dashed area (benchmark)


Step 6: Detailed Feature Comparison Table 📊


Numerical analysis with status indicators:


Your Value: Current song's feature value
High Pop Avg: Average of top 25th percentile songs
Difference: Absolute gap
Diff %: Percentage difference
Status: 🟢 Close (<10%), 🟡 Difference (10-20%), 🔴 Large Difference (>20%)


4. Feature Distribution Exploration Analysis 💻


Purpose: Explore feature relationships in the dataset


Features:


Interactive Hexbin Plot: Visualize feature correlation and distribution density
Customizable Axes: Select any two numeric features for X and Y axes
Correlation Display: Shows Pearson correlation coefficient (r value)
Joint Distribution: Marginal histograms on both axes


Use Case: Discover hidden patterns like "Do high-energy songs tend to be louder?" or "How does BPM relate to popularity?"


Dataset Information


Source: Spotify Top 2000s Mega Dataset from Kaggle


Size: 1,994 songs with 15 features


Features:


Audio Characteristics: BPM, Energy, Danceability, Loudness, Valence, Acousticness, Speechiness, Liveness, Duration
Metadata: Year, Artist, Genre, Title
Target Variable: Popularity Score (11-100)


Data Quality:


All features are numerical or can be encoded
No missing values in core features
Genre grouped into 4 major categories for modeling


Model Performance


Metric	Value	Description
R²	0.3025	Explains 30.25% of popularity variance
MAE	9.8946	Average prediction error ~10 points
RMSE	12.2407	Root mean squared error


Model Choice: Random Forest Regressor


Handles non-linear feature relationships
Provides feature importance ranking
Robust to outliers


Performance Context:
R²=0.3025 is meaningful for behavioral data where human preferences are influenced by external factors (marketing, social media, artist fame) beyond audio features. The model successfully identifies audio feature patterns that contribute to popularity.


Technical Highlights
SHAP Integration


Uses TreeExplainer for Random Forest interpretability
Visualizes feature contribution to individual predictions
Helps users understand model reasoning
NLTK Sentiment Analysis


Analyzes song title emotional tone
VADER lexicon optimized for social media text
Positive titles correlate with higher popularity
Plotly Interactive Charts


Radar chart with hover tooltips
Comparison table with color-coded status
Responsive design for all screen sizes
Dynamic Path Resolution


Uses os.path.dirname(os.path.abspath(__file__)) for file location
Works regardless of working directory
Clear error messages if files are missing


Notes


All .pkl and .csv files must be in the same directory as app.py
The application uses automatic path resolution to locate files
If Spotify-2000.csv is missing, download it from Kaggle
Model predictions are for reference only - external factors (marketing, timing, artist popularity) significantly influence actual song success


Author


ACC102 Course Project - Mini Assignment (Track 4: Interactive Data Tool)
