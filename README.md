# Crop Recommendation System

A machine-learning crop classification project with a Streamlit user interface.

## Files
- `Crop_Recommendation_System_Milestone.ipynb`: ML workflow and model training
- `app.py`: Streamlit frontend
- `crop_recommendation_model.joblib`: model selected in the latest notebook run
- `Crop_recommendation.csv`: cleaned dataset used in the latest run
- `model_results.csv`: evaluation metrics generated in the latest run
- `requirements.txt`: Python dependencies

## Run in Google Colab
1. Open the notebook and run all cells from top to bottom.
2. Run the final export cell to create the application files.
3. Install dependencies: `!pip install -r requirements.txt`
4. Start Streamlit: `!streamlit run app.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &`
5. In another cell run: `!npx localtunnel --port 8501`
6. Open the public URL printed by LocalTunnel. Keep the Colab runtime running.

## Run locally
1. Download all project files into the same folder.
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `streamlit run app.py`
4. Open the local URL shown in the terminal (usually http://localhost:8501).

The model metrics shown by the app come from the latest notebook run. No fixed performance values are hardcoded.
