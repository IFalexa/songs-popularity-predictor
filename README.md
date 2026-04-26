# Music Popularity Prediction Assistant

A data-driven tool that predicts song popularity based on audio features using machine learning, designed for independent music producers, A&R departments, and music marketing teams.

## Features

- **Data Model Performance Evaluation** - View R², MAE, RMSE metrics and prediction scatter plot
- **Feature Driver Factor Analysis** - Understand which audio features drive popularity predictions with tier analysis
- **Popularity Simulation Prediction** - Input your song features via sliders and get instant predictions with optimization suggestions
- **Feature Distribution Exploration** - Visualize relationships between different audio features with hexbin plots

## Tech Stack

- Python 3.9+
- pandas 2.3.3 - Data manipulation
- numpy 1.26.4 - Numerical computing
- scikit-learn 1.3.2 - Machine learning (Random Forest)
- matplotlib 3.10.8 - Visualization
- seaborn - Statistical visualization
- Streamlit 1.53.0 - Web application
- SHAP 0.49.1 - Model interpretation
- NLTK 3.8.1 - Sentiment analysis

## Quick Start for Instructors

Follow these steps to run the application locally after cloning the repository.

### Step 1: Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone https://github.com/IFalexa/songs-popularity-predictor.git
cd songs-popularity-predictor
```

Alternatively, download the ZIP file from GitHub by clicking "Code" → "Download ZIP".

### Step 2: Verify Python Version

This project requires Python 3.9 or higher. Check your Python version:

```bash
python --version
```

If you have multiple Python versions, use:

```bash
python3 --version
```

If Python is not installed, download it from https://www.python.org/downloads/

### Step 3: Create Virtual Environment (Recommended)

Creating a virtual environment avoids dependency conflicts:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt after activation.

### Step 4: Install Dependencies

Install all required packages:

```bash
pip install -r requirements.txt
```

This will install all dependencies including:
- streamlit==1.53.0
- pandas==2.3.3
- numpy==1.26.4
- scikit-learn==1.3.2
- matplotlib==3.10.8
- seaborn
- shap==0.49.1
- nltk==3.8.1
- joblib==1.5.3
- plotly
- And other supporting libraries

### Step 5: Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will automatically open in your default browser at http://localhost:8501

If the browser doesn't open automatically, manually navigate to the URL shown in the terminal.

## Troubleshooting & Common Issues

### Issue 1: "streamlit: command not found"

**Solution:** Install Streamlit globally or use the full path:

```bash
python -m streamlit run app.py
```

### Issue 2: ModuleNotFoundError for specific packages

**Solution:** Install the missing package individually:

```bash
pip install streamlit
pip install pandas
pip install scikit-learn
pip install shap
pip install nltk
```

### Issue 3: SHAP import error on Windows

**Solution:** SHAP requires Visual C++ Build Tools. Install it from:
https://visualstudio.microsoft.com/visual-cpp-build-tools/

Or use a pre-compiled version:

```bash
pip install shap --no-binary shap
```

### Issue 4: Python version compatibility

**Minimum Required:** Python 3.9

If you encounter compatibility issues with newer Python versions (3.11+), try:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Issue 5: Permission denied when saving model files

**Solution:** Close any programs that might be using the .pkl files, or save with a different filename:

```python
joblib.dump(model, 'model_new.pkl', compress=3)
```

### Issue 6: NLTK data not found

**Solution:** Download required NLTK data:

```python
import nltk
nltk.download('vader_lexicon')
```

### Issue 7: Virtual environment activation fails on Windows

**Solution:** Enable PowerShell script execution:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating again:

```bash
venv\Scripts\activate
```

### Issue 8: Plotly visualization not rendering

**Solution:** Ensure plotly is installed:

```bash
pip install plotly
```

## Project Structure

- `app.py` - Main Streamlit application with 4 interactive modules
- `train_model.py` - Model training script
- `Music Popularity Prediction.ipynb` - Complete Jupyter Notebook with data analysis workflow
- `music_popularity_rf.pkl` - Trained Random Forest model
- `feature_columns.pkl` - Feature column names used in training
- `test.pkl` - Test data for model evaluation
- `Spotify-2000.csv` - Dataset from Kaggle (1,994 songs with 15 features)
- `requirements.txt` - Python dependencies with versions
- `README.md` - This documentation file

## Model Performance

- R² Score: 0.325 (explains 32.5% of variance in popularity)
- Mean Absolute Error (MAE): 9.73 (predictions within ~10 popularity points on average)
- Root Mean Square Error (RMSE): 12.04

Top 3 Feature Importance:
- Release Year: 11.8% (era characteristics)
- Song Duration: 9.0% (streaming-era preference for shorter songs)
- Dutch Indie Genre: 8.5% (regional market patterns)

## Dataset Information

The project uses the Spotify Top 2000s Mega Dataset from Kaggle, containing:
- 1,994 songs
- 15 features including:
  - Audio characteristics: BPM, Energy, Danceability, Loudness, Valence, Acousticness, Speechiness, Liveness, Duration
  - Metadata: Year, Artist, Genre, Title
- Time range: 1956-2019
- Target variable: Popularity score (11-100)

Dataset source: https://www.kaggle.com/datasets/iamsumat/spotify-top-2000s-mega-dataset

## Application Modules

### 1. Data Model Performance Evaluation
- View R², MAE, RMSE metrics with status indicators
- Actual vs. Predicted scatter plot with diagonal reference line
- Deep analysis conclusions on model performance

### 2. Feature Driver Factor Analysis
- Global influence weight bar chart of all features
- Three-tier feature categorization:
  - First Tier: Era Characteristics (Year)
  - Second Tier: Rhythm Dynamics (Danceability, Loudness, BPM, Valence)
  - Third Tier: Market Labels (Genre patterns)

### 3. Popularity Simulation Prediction
- Interactive parameter configuration with sliders
- Song name, duration, genre selection
- Audio feature adjustments (Danceability, Energy, Valence, etc.)
- Real-time popularity prediction with status indicator
- Optimization suggestions based on high-popularity benchmarks
- SHAP Force Plot for feature attribution
- Radar chart comparison with high-popularity songs
- Detailed feature value comparison table

### 4. Feature Distribution Exploration
- Hexbin plots for feature relationship visualization
- Interactive X-axis and Y-axis selection
- Correlation coefficient display
- Distribution histograms for both axes

## Running the Jupyter Notebook

To explore the data analysis process:

```bash
jupyter notebook "Music Popularity Prediction.ipynb"
```

Or if using JupyterLab:

```bash
jupyter lab "Music Popularity Prediction.ipynb"
```

The notebook contains:
- Problem Definition & Business Context
- Data Overview & Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Building with GridSearchCV
- Model Evaluation & Feature Importance Analysis
- Business Insights & Product Design
- Limitations & Future Work

## System Requirements

- Operating System: Windows 10+, macOS 10.14+, or Linux
- Python: 3.9 or higher (tested on Python 3.9)
- RAM: 4GB minimum (8GB recommended)
- Disk Space: 500MB for dependencies

## Additional Resources

- Streamlit Documentation: https://docs.streamlit.io/
- scikit-learn Documentation: https://scikit-learn.org/stable/
- SHAP Documentation: https://shap.readthedocs.io/
- Dataset Source: https://www.kaggle.com/datasets/iamsumat/spotify-top-2000s-mega-dataset

## Author

Created for ACC102 Mini Assignment

## License

This project is for educational purposes only.

## Acknowledgments

- Dataset source: Kaggle Spotify Top 2000s Mega Dataset
- Model: Random Forest Regression with GridSearchCV hyperparameter tuning
- Visualization: Streamlit, matplotlib, seaborn, SHAP, plotly
