# Music Popularity Prediction Assistant

A machine learning-based tool that predicts song popularity using audio features and provides actionable optimization suggestions for music creators.

## Demo

![Model Performance](demo_1.png)

![Feature Importance](demo_2.png)

![SHAP Analysis](demo_3.png)

![Radar Chart](demo_4.png)

![Comparison Table](demo_5.png)

![Feature Distribution](demo_6.png)

## Project Overview

This project develops a Random Forest regression model to predict song popularity scores based on audio features from Spotify's Top 2000s dataset. The trained model is deployed as a Streamlit application with real-time prediction and benchmarking capabilities.

**Target Users**: Independent music producers, A&R departments, music marketing teams

**Core Features**:
- Popularity prediction based on audio features
- Optimization suggestions with benchmark gaps
- Feature comparison via radar charts

## Dataset

**Source**: [Spotify Top 2000s Mega Dataset](https://www.kaggle.com/datasets/iamsumat/spotify-top-2000s-mega-dataset)

**Records**: 1,994 songs

**Features**: 15 columns including:
- Audio features: BPM, Energy, Danceability, Loudness, Valence, Acousticness, Speechiness, Liveness
- Metadata: Year, Genre, Artist, Title
- Target: Popularity (0-100 scale)

## Methodology

### Tech Stack
- Python 3.x
- scikit-learn (Random Forest Regression)
- pandas, numpy
- matplotlib, seaborn
- Streamlit
- SHAP

### Data Processing Pipeline
1. Data Cleaning: Handle missing values, duplicates, outliers
2. Feature Engineering: Artist encoding, Genre grouping, Title length extraction
3. Train/Test Split: 80/20 ratio
4. Model Training: Random Forest with optimized hyperparameters
5. Evaluation: R², MAE, RMSE metrics

### Why Random Forest?
- Captures non-linear relationships in audio features
- Resistant to overfitting compared to single decision trees
- Provides interpretable feature importance scores
- Outperformed simpler models (Linear Regression, Decision Tree) in cross-validation

## Model Performance

**R² Score**: 0.325

**MAE**: 9.74

**RMSE**: 12.04

**Interpretation**: The model explains 32.5% of popularity variance, with predictions averaging ±10 points error. RMSE > MAE indicates presence of prediction outliers - some songs with average features became viral hits, while others with strong features underperformed.

### Feature Importance (Top 10)

1. **Year** (11.8%) - Temporal trends and streaming era effects
2. **Length/Duration** (9.0%) - Shorter songs preferred in streaming era
3. **Genre: Dutch Indie** (8.5%) - Regional bias indicator
4. **Danceability** (7.5%) - Rhythm and groove importance
5. **Liveness** (7.5%) - Live performance presence
6. **Loudness** (7.4%) - Production quality indicator
7. **BPM** (7.1%) - Tempo impact
8. **Valence** (6.6%) - Emotional positivity
9. **Title Length** (6.6%) - Memorability factor
10. **Energy** (6.2%) - Song energy level

**Key Insights**:
- Release year is the strongest predictor (11.8% importance)
- Genre plays significant role, especially niche genres
- Audio features (Danceability, Loudness, BPM) collectively drive popularity
- Duration optimization: shorter songs preferred in streaming era

## Streamlit Application

The application consists of 4 interactive modules:

### 1. Model Performance Dashboard
- Display R², MAE, RMSE metrics
- Model evaluation visualizations
- Performance comparison charts

### 2. Feature Driver Analysis
- Feature importance ranking
- SHAP analysis for interpretability
- Detailed explanation of how each feature impacts predictions

### 3. Popularity Prediction
- Input audio features via sliders (BPM, Energy, Danceability, etc.)
- Get instant popularity score prediction
- Compare with top songs benchmark
- Receive optimization suggestions

### 4. Feature Distribution
- Visualize feature distributions across popularity levels
- Identify optimal feature ranges for high popularity
- Benchmark your song against successful tracks

## Quick Start

### Prerequisites

```bash
pip install streamlit pandas numpy scikit-learn matplotlib seaborn shap
```

### Run Application

```bash
streamlit run app.py
```

### Use Pre-trained Model

The repository includes a pre-trained Random Forest model (`music_popularity_rf.pkl`) that can be loaded directly:

```python
import pickle
with open('music_popularity_rf.pkl', 'rb') as f:
    model = pickle.load(f)
```

## Model Details

**Algorithm**: Random Forest Regressor

**Hyperparameters**:
- n_estimators: 350
- max_depth: 14
- min_samples_leaf: 2
- max_features: None
- random_state: 42

**Training Data**: 1,595 songs (80%)

**Test Data**: 399 songs (20%)

## Limitations

**Data Limitations**:
- Limited to songs from 1956-2019
- Missing contemporary features (streams, playlists, social media metrics)
- Genre distribution skewed towards certain categories
- Does not include lyrics or music video features

**Model Limitations**:
- R² of 0.325 indicates significant unexplained variance
- Cannot capture viral/meme-driven popularity
- Regional popularity differences not considered
- Artist reputation and marketing budget not factored

## Future Improvements

**Short-term**:
- Add more recent songs (2020-2024)
- Include streaming platform features (playlists, skip rates)
- Experiment with ensemble models (XGBoost, LightGBM)
- Add cross-validation for more robust evaluation

**Long-term**:
- Deploy as web service API
- Add real-time Spotify integration
- Incorporate lyrical analysis (NLP)
- Include artist popularity and social media metrics
- Build recommendation engine based on similarity

## Project Structure

```
music-popularity-prediction/
├── app.py
├── music_popularity_rf.pkl
├── songs-popularity-prediction.ipynb
├── README.md
├── demo_1.png
├── demo_2.png
├── demo_3.png
├── demo_4.png
├── demo_5.png
└── demo_6.png
```

**File Descriptions**:
- `app.py` - Streamlit application with 4 interactive modules
- `music_popularity_rf.pkl` - Pre-trained Random Forest model
- `songs-popularity-prediction.ipynb` - Complete Jupyter notebook with data analysis and model training
- `README.md` - Project documentation (this file)
- `demo_1.png` - Model Performance Dashboard screenshot
- `demo_2.png` - Feature Importance visualization
- `demo_3.png` - SHAP Analysis visualization
- `demo_4.png` - Radar Chart for feature comparison
- `demo_5.png` - Comparison Table with benchmark
- `demo_6.png` - Feature Distribution analysis

## Ethical Considerations

- Model predictions should guide, not dictate creative decisions
- Popularity metrics do not equal artistic quality
- Genre bias exists in training data
- Transparency in model limitations communicated to users
- Avoid over-reliance on algorithmic predictions for artistic choices

## AI Disclosure

This project utilized AI tools (Claude, ChatGPT) for:
- Code debugging and optimization
- Documentation writing assistance
- Exploratory data analysis support

All modeling decisions, interpretations, and final code were reviewed and approved by the human author.

## Requirements

```
streamlit>=1.20.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
shap>=0.41.0
```

## License

This project is for educational purposes as part of ACC102 coursework.

---

**Questions or Feedback**: Please open an issue in this repository.
