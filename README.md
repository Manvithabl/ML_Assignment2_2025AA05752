**a.	Problem statement**
This project is to build and compare different machine learning models to predict whether a breast tumour is malignant (cancerous) or benign (non-cancerous). 6 classification algorithms are implemented, and an interactive Stream lit web application is developed to allow data to be uploaded, a model to be selected, and prediction results along with evaluation metrics to be displayed.
**b.	Dataset description**
The dataset used for this project is the Breast Cancer Classification dataset obtained from Kaggle. It is a binary classification dataset containing 569 patient records with 30 numerical features.
Each record describes characteristics of cell nuclei, such as radius, texture, perimeter, and area. The target variable is diagnosis, where:
•	M represents malignant (encoded as 1)
•	B represents benign (encoded as 0)
Before training the models, unnecessary columns were removed, the target variable was encoded, missing values were handled using mean imputation, and feature scaling was applied where required. The dataset was then split into training (80%) and testing (20%) sets for evaluation.
**c. Models Used**

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|------|-----------|--------|------|------|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |
| kNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Random Forest | 0.9737 | 0.9929 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| XGBoost | 0.9737 | 0.9940 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

**Observations on the Performance of Each Model**
| ML Model Name | Observation about Model Performance |
|---------------|--------------------------------------|
| Logistic Regression | Performed consistently well across all evaluation metrics, showing strong class separation and a good balance between precision and recall. This indicates that the dataset is largely linearly separable and suitable for linear classification methods. |
| Decision Tree | Delivered comparatively lower overall performance among the models. While it maintained balanced precision and recall, its generalization ability was weaker than the other classifiers, which may be due to its tendency to overfit. |
| kNN | Showed strong predictive capability with high precision and competitive overall performance. However, its recall was slightly lower compared to the best-performing models, indicating some missed positive cases. |
| Naive Bayes | Achieved good class discrimination but demonstrated relatively lower recall and overall balance compared to other methods. The feature independence assumption may have limited its performance on this dataset. |
| Random Forest (Ensemble) | One of the best-performing models with strong overall consistency across all evaluation metrics. The ensemble approach improved stability and reduced overfitting. |
| XGBoost (Ensemble) | Matched the top performance among all models. It effectively captured complex relationships in the data and demonstrated excellent overall balance between precision and recall, making it one of the most reliable models in this project. |
