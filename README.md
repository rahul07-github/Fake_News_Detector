# 📰 Fake News Detector

This project applies Machine Learning techniques to detect and classify news articles as fake or real. It provides an interactive interface (via Jupyter Notebook and web app) to analyze news content, view dataset statistics, and predict the authenticity of user-provided text.

# 🙌 Acknowledgment

- Dataset inspired by open-source fake news detection challenges (e.g., Kaggle, LIAR dataset).

- Libraries used: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.

- Special thanks to the open-source community for datasets, frameworks, and visualization tools.

  --------
  # 🚀 How to Use

## 1.Open the Notebook

- Launch Jupyter Notebook:
  jupyter notebook
2. Load the Dataset

The dataset (train.csv, test.csv, etc.) can be loaded with Pandas:
import pandas as pd
data = pd.read_csv("data/train.csv")
3.Run the Workflow

- Preprocessing: Cleans news text (removes stopwords, punctuation, etc.).

- Feature Extraction: Converts text into numerical form using TF-IDF Vectorization.

- Model Training: Trains models such as Random Forest, Logistic Regression, or Naive Bayes.

- Evaluation: Reports accuracy, precision, recall, and F1-score.

- Prediction: Input your own text and get predictions with confidence scores

------------
# 📂 Project Structure
Fake_News_Detector/
│
├── data/
│   ├── train.csv             # Training dataset
│   ├── test.csv              # Test dataset
│   └── submission.csv        # Sample submission
│
├── notebooks/
│   └── Face News Prediction Model.ipynb   # Main notebook
│
├── images/                   # App screenshots & visualizations
│   ├── About.png
│   ├── Data Stats.png
│   └── Predict.png
│
├── requirements.txt          # Required dependencies
└── README.md                 # Documentation

-----------------
# 📊 Features

- Fake News Detection → Detect whether an article is Real or Fake.

- Multiple Labels → Predictions include categories like True, Mostly True, Half True, Barely True, Fake, Pants on Fire.

- Data Statistics → Pie charts, bar graphs, and text length distributions to understand the dataset.

- Confidence Levels → Model outputs probability scores for predictions.

# ⚙️ Requirements

Install dependencies before running the notebook:
      pip install -r requirements.txt

---------------
## Key dependencies:

- Python 3.7+

- Jupyter Notebook

- Pandas, NumPy

- Scikit-learn

- Matplotlib, Seaborn

# 📬 Contact

For questions, suggestions, or contributions:

- Author: Rahul Kumar Jha

- Email: rahulkumarjha9643@gmail.com

- GitHub: https://github.com/rahul07-github/Fake_News_Detector.git
