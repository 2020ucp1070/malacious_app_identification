# 🛡️ Malicious App Identification Using Machine Learning

This project focuses on identifying malicious Android applications using Machine Learning models. It detects malware, spyware, and adware apps based on extracted features like permissions, intents, API calls, and manifest attributes.

Built as a final-year project at MNIT Jaipur, this work creates a full end-to-end pipeline — from data generation to model training, evaluation, and comparison — to build a robust malware detection system.

The project compares multiple ML algorithms to find the best performing model for real-world cybersecurity applications.

---

## 📂 Project Architecture

malacious_app_identification/
│
├── FINAL_YEAR_PROJECT_FINAL.ipynb               # Main notebook for consolidated analysis
├── Final_Major_Project_MA_random_forest.ipynb    # Random Forest model notebook
├── Final_Major_Project_MA_knn.ipynb              # K-Nearest Neighbors model notebook
├── Final_Major_Project_MA_ada_boost.ipynb        # AdaBoost Classifier notebook
├── Final_Major_Project_MA_mlp.ipynb              # MLP Classifier notebook
├── Final_Major_Project_MA_logistic_regression.ipynb # Logistic Regression notebook
├── Final_Major_Project_MA_naive_bayes.ipynb      # Naive Bayes model notebook
├── Final_Major_Project_MA_lda.ipynb              # Linear Discriminant Analysis notebook
├── Final_Major_Project_MA_decision_tree.ipynb    # Decision Tree model notebook
│
├── app.py         # Main logic to execute ML models
├── checker.py     # Check extracted features
├── compare.py     # Compare model performances
├── convert.py     # Convert datasets to required format
├── merge.py       # Merge multiple datasets
│
└── README.md      # Project documentation

---

## ⚙️ Installation Instructions

# 1. Clone the repository
git clone https://github.com/2020ucp1070/malacious_app_identification.git
cd malacious_app_identification

# 2. (Optional) Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install required dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 4. Launch Jupyter Notebook
jupyter notebook

# 5. Open and run any model file (e.g., Final_Major_Project_MA_random_forest.ipynb)

---

## 🧠 Machine Learning Models Implemented

- Random Forest Classifier
- Decision Tree Classifier
- Logistic Regression
- Naive Bayes Classifier
- K-Nearest Neighbors (KNN)
- Multi-layer Perceptron (MLP)
- Linear Discriminant Analysis (LDA)
- AdaBoost Classifier

Each model is trained on the generated malicious/benign app dataset and evaluated on multiple performance metrics.

---

## 🗃️ Dataset Description

- Collected APK feature data: permissions, intents, activities, services.
- Labeled apps manually into benign, adware, malware.
- Created structured datasets after feature extraction.
- Preprocessing included normalization and handling missing data.

---

## 📊 Results and Evaluation

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Random Forest        | 96.5%    | 95.3%     | 97.1%  | 96.2%    |
| Decision Tree        | 93.2%    | 92.0%     | 94.5%  | 93.2%    |
| Logistic Regression  | 89.6%    | 88.5%     | 90.2%  | 89.3%    |
| MLP Classifier       | 94.1%    | 92.9%     | 95.1%  | 94.0%    |

📈 Random Forest and MLP Classifier achieved the highest performance with strong generalization on unseen data.

---

## 🎯 Key Takeaways

- Machine Learning can successfully detect malicious Android apps with high accuracy.
- Random Forest and MLP performed best among the models compared.
- Developed a complete modular pipeline for malware app classification.
- Practical solution for improving Android app security using data-driven methods.

---

## 📩 Contact Information

👩‍💻 Author: Anitha D 
📧 Email: danitha200204@gmail.com 
🔗 LinkedIn: https://www.linkedin.com/in/d-anitha-990806237/  
🐙 GitHub: https://github.com/2020ucp1070?tab=repositories

---

## 📜 License

This project is licensed under the MIT License.  
You are free to use, modify, and distribute this work with proper attribution.

---

