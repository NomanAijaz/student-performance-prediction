# 🎓 Student Performance Prediction System

A comprehensive machine learning project that predicts student final exam scores (G3) based on demographic, social, and academic factors. This is a **learning-focused project** implementing linear regression from scratch and exploring the complete ML pipeline.

## 📊 Project Overview

**Objective**: Predict final Math grades (G3) for Portuguese secondary school students using 32 features including demographics, family background, study habits, social factors, and previous grades (G1, G2).

**Dataset**: Student Performance Dataset from UCI ML Repository
- **Students**: ~395
- **Features**: 32 (30 attributes + G1 + G2)
- **Target**: G3 (final grade, 0-20 scale)
- **Subject**: Mathematics

**Approach**: Phase-by-phase implementation with emphasis on understanding concepts, not just coding.

---

## 🗂️ Project Structure

```
student-performance-prediction/
│
├── data/
│   ├── raw/                    # Original dataset
│   ├── processed/              # Cleaned and preprocessed data
│   └── README.md              # Dataset documentation
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda_visualization.ipynb
│   ├── 03_data_preprocessing.ipynb
│   ├── 04_linear_regression_scratch.ipynb
│   ├── 05_gradient_descent_deep_dive.ipynb
│   ├── 06_model_evaluation.ipynb
│   ├── 07_feature_engineering.ipynb
│   └── 08_final_model_deployment.ipynb
│
├── src/
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessor.py        # Data cleaning functions
│   ├── linear_regression.py   # Custom implementation
│   ├── visualizer.py          # Visualization functions
│   ├── evaluator.py           # Model evaluation
│   └── predictor.py           # Prediction interface
│
├── models/
│   └── trained_models/        # Saved models
│
├── reports/
│   └── figures/               # Generated plots and visualizations
│
├── docs/
│   ├── phase_studies/         # Study documents for each phase
│   ├── phase_conclusions/     # Conclusion documents for each phase
│   └── Project_description.md # Complete project guide
│
├── .cursorrules               # Project workflow rules
├── requirements.txt           # Python dependencies
├── .gitignore                # Git ignore rules
└── README.md                 # This file
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda package manager
- Jupyter Notebook

### Installation

1. **Clone the repository**:
```bash
git clone git@github.com:NomanAijaz/student-performance-prediction.git
cd student-performance-prediction
```

2. **Create virtual environment** (recommended):
```bash
# Using conda
conda create -n student-perf python=3.10
conda activate student-perf

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**:
```bash
jupyter notebook
```

---

## 📚 Project Phases

This project follows a structured, phase-by-phase approach:

### ✅ Phase 0: Initial Setup (Current)
- Project structure created
- Dataset loaded
- Documentation prepared

### 📍 Phase 1: Data Collection & Understanding (Next)
- Load and inspect dataset
- Understand features and target
- Check data quality
- Document initial findings

### Phase 2: Exploratory Data Analysis
- Statistical analysis
- Visualizations
- Correlation analysis
- Feature relationships

### Phase 3: Data Preprocessing
- Handle missing values
- Outlier detection and treatment
- Feature scaling
- Train-test split

### Phase 4: Linear Regression from Scratch
- Implement simple linear regression
- Implement multiple linear regression
- Understand cost function
- Compare with scikit-learn

### Phase 5: Gradient Descent Deep Dive
- Experiment with learning rates
- Implement optimization algorithms
- Visualize convergence
- Performance comparison

### Phase 6: Model Evaluation
- Calculate regression metrics
- Residual analysis
- Feature importance
- Error analysis

### Phase 7: Feature Engineering & Selection
- Create new features
- Feature selection methods
- Regularization (Ridge, Lasso)
- Model comparison

### Phase 8: Model Optimization & Deployment
- Hyperparameter tuning
- Cross-validation
- Final model selection
- Build prediction interface

---

## 📖 Learning Approach

Each phase follows this cycle:

1. **📚 STUDY**: Read phase study document (`docs/phase_studies/`)
2. **📝 DOCUMENT**: Understand concepts deeply with mathematical foundations
3. **💻 IMPLEMENT**: Code the phase in Jupyter notebook
4. **🧪 TEST**: Validate implementation works correctly
5. **✅ CONCLUDE**: Write conclusion document (`docs/phase_conclusions/`)
6. **➡️ PROCEED**: Move to next phase

**Philosophy**: Understanding over speed. This is a learning project, not a race.

---

## 🎯 Learning Objectives

By completing this project, you will:

- ✅ Master linear regression theory and implementation
- ✅ Understand gradient descent optimization
- ✅ Perform comprehensive EDA
- ✅ Handle real-world data challenges
- ✅ Implement ML algorithms from scratch
- ✅ Evaluate and interpret model performance
- ✅ Engineer and select features
- ✅ Build end-to-end ML pipeline

---

## 📊 Dataset Information

**Source**: UCI Machine Learning Repository  
**Citation**: P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. 2008.

**Key Features**:
- Demographics: age, sex, address, family size
- Family: parent education, jobs, relationships
- Academic: study time, failures, support, absences
- Social: going out, alcohol consumption, relationships
- Grades: G1 (1st period), G2 (2nd period), G3 (final - TARGET)

See `data/README.md` for complete attribute descriptions.

---

## 🛠️ Technologies Used

- **Python 3.10+**
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Matplotlib & Seaborn**: Visualization
- **Scikit-learn**: ML algorithms and metrics
- **Jupyter**: Interactive development
- **Streamlit**: Web interface (Phase 8)

---

## 📈 Current Status

- **Phase**: 0 - Initial Setup ✅
- **Next**: Phase 1 - Data Collection & Understanding
- **Progress**: 0/8 phases complete

---

## 🤝 Contributing

This is a personal learning project, but feedback and suggestions are welcome!

---

## 📄 License

This project is for educational purposes.

---

## 👤 Author

**Noman Aijaz**  
GitHub: [@NomanAijaz](https://github.com/NomanAijaz)

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the dataset
- Project structure inspired by ML best practices
- Learning resources from Andrew Ng's ML course

---

**Ready to start learning? Begin with Phase 1!** 🚀

Read the study document: `docs/phase_studies/Phase_01_Data_Collection_Understanding.md`
