# 📰 Fake News Detection using Machine Learning

This project implements a **Fake News Detection** system using **Natural Language Processing (NLP)** and **Machine Learning** techniques.  
The system classifies a given news article as **Fake** or **Real** based on its textual content.

A simple and interactive **Streamlit** web application is used to perform real-time predictions.

---

## 📌 Problem Statement

With the rapid growth of digital media, misinformation and fake news spread quickly across platforms.  
Manual verification is time-consuming and error-prone.  

This project aims to automatically detect fake news using supervised machine learning models trained on robust datasets.

---

## 🧠 Methodology

1. Collected separate **Fake** and **Real** news datasets.  
2. Added class labels and merged them into a single dataset.  
3. Enhanced dataset robustness through careful data balancing and cleaning.  
4. Performed text preprocessing.  
5. Converted text data into numerical features using **TF-IDF Vectorization**.  
6. Trained a **Linear Support Vector Machine (SVM)** classifier.  
7. Saved the trained **model** and **vectorizer**.  
8. Built a **Streamlit app** for inference and deployment.

---

## 🧩 Dataset Robustness

To improve model generalization and reduce bias:

- Ensured balanced representation of Fake and Real news articles.  
- Removed duplicate and near-duplicate entries.  
- Cleaned noisy text (URLs, extra symbols, irrelevant tokens).  
- Verified label consistency after merging datasets.  
- Added diverse news samples to prevent overfitting on specific patterns.  

These steps help the model perform better on **unseen** and **real-world** news articles rather than memorizing dataset-specific patterns.

---

## 🛠️ Tech Stack

**Language:** Python  

**Libraries:**
- scikit-learn  
- pandas  
- numpy  
- streamlit  

**Model:** Linear Support Vector Machine (SVM)  
**Text Vectorization:** TF-IDF  
**Deployment Platform:** Streamlit Community Cloud

---

## 📂 Project Structure
```
LEVEL1_FakeNewsDetection/
│
├── app.py  #Streamlit UI for checking prediction
│
├── models/
│   ├── svm_fakeNews_model.pkl   # Trained Balanced Linear SVM model
│   └── fakeNews_vectorizer.pkl   # TF-IDF vectorizer
│
├── requirements.txt
├── README.md
```

---

## 🚀 Deployment

The application is deployed using **Streamlit Community Cloud**, allowing seamless hosting of the Streamlit application directly from GitHub.

**Deployment Process:**
1. Push the complete project to a public GitHub repository.  
2. Add `requirements.txt` for dependency management.  
3. Connect the repository to **Streamlit Community Cloud**.  
4. Deploy `app.py` as the entry point.

**Live Application:**  
🔗 App URL: https://level1fakenewsdetection-a6qkexh7yxmvsh7hh7wuyw.streamlit.app/

---

## 🖥️ Application Workflow

1. User enters a news article into the text input field.  
2. Text is transformed using a pre-trained **TF-IDF vectorizer**.  
3. The **Linear SVM** model predicts whether the news is **Fake** or **Real**.  
4. A confidence score is displayed indicating the model’s certainty.  

> The confidence score is derived from the SVM decision function and is **not a probability value**.

---

## 📊 Model Information

- **Algorithm:** Linear Support Vector Machine (SVM)  
- **Feature Extraction:** TF-IDF  
- **Classes:**  
  - `0` → Fake News  
  - `1` → Real News  

> Model training and evaluation were performed offline.

---

## 📎 Dataset

The dataset consists of labeled **Fake** and **Real** news articles merged into a single dataset.  
Additional preprocessing steps were applied to improve **robustness** and **reliability**.
After augmenting the dataset in file named **hard_samples.csv** with 136 adversarial hard samples, model accuracy decreased marginally while robustness improved. The classifiers became less sensitive to stylistic cues and demonstrated more balanced error behaviour, indicating better generalization to real-world deceptive news.

---

## 🔮 Future Scope

- Add explainability using **LIME** or **SHAP**.  
- Expand dataset to support **multilingual fake news detection**.  
- Improve UI with optional **analytics** and visual insights.

---

## 👩‍💻 Author

**Harichandana**
Machine Learning and Python enthusiast

---

## 📜 License

This project is intended for **educational and academic purposes**.
