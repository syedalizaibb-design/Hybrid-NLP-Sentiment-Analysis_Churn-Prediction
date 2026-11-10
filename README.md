Hybrid NLP Architecture for Sentiment Analysis and Churn Prediction
Overview
This project explores natural language processing (NLP) for analyzing e-commerce customer reviews. The current implementation, SentiMart, is a Streamlit application that classifies reviews as Positive, Negative, or Neutral using a trained Support Vector Machine (SVM) and TF-IDF text features.
Project status: Sentiment analysis is implemented. Customer churn prediction is a proposed extension and is not implemented in the current uploaded application or training notebook.
Features
Analyze an individual customer review in the Streamlit interface.
Classify reviews into positive, negative, and neutral sentiment.
Display a suggested automatic response based on sentiment.
Upload a CSV file to analyze multiple reviews.
View sentiment distribution charts, sentiment proportions, and a word cloud.
Train and compare Logistic Regression, Linear SVM, and a multilayer perceptron (MLP) in the notebook.
Technologies
Python, Streamlit, pandas, scikit-learn, NLTK, Matplotlib, WordCloud, TF-IDF, Linear SVM, Logistic Regression, and MLPClassifier.
Project Structure
```text
Hybrid-NLP-Sentiment-Analysis-Churn-Prediction/
├── app.py
├── Model_Training.ipynb
├── train_data.csv
├── svm_model.pkl
├── tfidf_vectorizer.pkl
├── requirements.txt
├── README.md
└── .gitignore
```
Setup and Run
Clone the repository and enter its folder:
```bash
   git clone https://github.com/YOUR_USERNAME/Hybrid-NLP-Sentiment-Analysis-Churn-Prediction.git
   cd Hybrid-NLP-Sentiment-Analysis-Churn-Prediction
   ```
Create and activate a virtual environment (Windows):
```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
Install dependencies:
```bash
   pip install -r requirements.txt
   ```
Ensure `svm_model.pkl` and `tfidf_vectorizer.pkl` are in the same folder as `app.py`.
Start the application:
```bash
   streamlit run app.py
   ```
Open the local address displayed by Streamlit, usually `http://localhost:8501`.
Training Notebook
`Model_Training.ipynb` reads `train_data.csv`, maps sentiment labels to `-1`, `0`, and `1`, preprocesses review text with NLTK, creates TF-IDF features, and trains Logistic Regression, Linear SVM, and MLP classifiers. It saves the trained SVM and TF-IDF vectorizer as pickle files.
Before running the notebook: Replace its original Windows-specific dataset path with:
```python
df = pd.read_csv('train_data.csv')
```
Install NLTK if needed (`pip install nltk`). The notebook downloads its required NLTK resources.
Input Data
The included training dataset has review text in `reviews.text` and sentiment labels in `sentiment`. The batch analytics page looks for a `reviews.text` column in uploaded CSV files; if it is absent, the app currently uses the first column.
Limitations and Future Work
Churn prediction is not yet implemented. Future work could add customer-level behavioral data, churn labels, and a dedicated churn prediction model.
The notebook uses stopword removal and lemmatization during training, while the app currently uses simpler text cleaning. Align preprocessing for more reliable predictions.
Evaluate and document model performance on a held-out test set before publishing accuracy claims.
Load `.pkl` model files only from trusted sources.
Author
Add your name and GitHub profile here.
License
No license has been selected yet. Add a `LICENSE` file if you want to specify reuse permissions.
