# Hotel Reservation Classification using Random Forest

This project demonstrates a full machine learning workflow to classify hotel reservation statuses (confirmed or canceled) using supervised learning techniques. The main algorithm used is the **Random Forest Classifier**, with additional experimentation using **Logistic Regression**.

---

## Project Overview

The objective is to predict whether a hotel reservation will be honored or canceled based on features such as lead time, room type, meal plan, booking source, and more. The workflow includes:

- Exploratory Data Analysis (EDA)
- Outlier detection (IQR method)
- Feature engineering
- Categorical encoding (LabelEncoder + OneHotEncoding)
- Feature selection (based on importance scores)
- Model training (Random Forest, Logistic Regression)
- Hyperparameter tuning (GridSearchCV)
- Evaluation metrics: Accuracy, Confusion Matrix, Classification Report, ROC-AUC

---

## Files

- `hotel_reservations.ipynb`: Full Jupyter notebook with code and plots.
- `hotel_reservations.csv`: The dataset used (original).
- `README.md`: This documentation file.

---

## Dataset Info

The dataset includes fields such as:

- `lead_time`: Days between booking and arrival
- `avg_price_per_room`: Price per night
- `meal_plan`, `room_type_reserved`, `market_segment_type`: Categorical variables
- `no_of_week_nights`, `no_of_weekend_nights`: Length of stay
- `repeated_guest`, `booking_status`: Target variable

Additional engineered features include:

- `trusted`: Indicates guest reliability based on past behavior
- `total_nights`: Sum of week and weekend nights

---

## Model Training

### Random Forest Classifier

- Importance-based feature selection
- Hyperparameter tuning with `GridSearchCV`
- Evaluation using accuracy, confusion matrix, ROC-AUC

### Logistic Regression

- Regularization (`C`) and solver tuning
- Benchmark for comparison with Random Forest

---
