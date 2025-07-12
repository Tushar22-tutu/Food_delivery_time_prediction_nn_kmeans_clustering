# 🍽️ Food Delivery Time Prediction

## 🧠 Objective

**Predict whether a food delivery will be _Fast_ or _Delayed_** using machine learning techniques.  
Based on features like:

- Customer location
- Restaurant location
- Weather conditions
- Traffic conditions
- Order priority

This project includes **Clustering (K-Means, Hierarchical)** and a **Neural Network-based classification model**.

---

## 📊 Phase 1 - Data Preprocessing and Feature Engineering

### 🔹 Data Cleaning

- Load the dataset: `Food_Delivery_Time_Prediction.csv`
- Handle missing values (imputation/removal)
- Encode categorical variables using **One-Hot Encoding** or **Label Encoding**
- Normalize numeric features like `Distance` and `Delivery Time`

### 🔹 Feature Engineering

- Calculate **Geographical Distance** using Haversine formula (if not already present)
- Add **Time-based Features** like `Rush Hour` or `Weekend Indicator`

---

## 🔍 Phase 2 - Clustering

### ✅ K-Means Clustering

**Objective:** Cluster similar deliveries based on common features

**Steps:**
- Use Elbow Method to find `optimal clusters`
- Train and visualize clusters

### ✅ Hierarchical Clustering

- Use **Agglomerative Clustering**
- Visualize using **Dendrogram**
- Compare with K-Means clusters

---

## 🤖 Phase 3 - Neural Networks

### 🔹 Architecture

```python
model = Sequential()
model.add(Dense(64, activation='relu', input_dim=X.shape[1]))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
