# Phase 1: Data Collection & Understanding - Study Guide

## Table of Contents
1. [Introduction to Data Collection in Machine Learning](#1-introduction)
2. [Understanding Our Dataset](#2-understanding-our-dataset)
3. [Feature Types and Their Importance](#3-feature-types)
4. [Statistical Foundations](#4-statistical-foundations)
5. [Data Quality Concepts](#5-data-quality-concepts)
6. [Target Variable Understanding](#6-target-variable)
7. [Relationships Between Features](#7-feature-relationships)
8. [Mathematical Notation](#8-mathematical-notation)
9. [Practical Considerations](#9-practical-considerations)
10. [Phase 1 Implementation Goals](#10-implementation-goals)

---

## 1. Introduction to Data Collection in Machine Learning {#1-introduction}

### What is Data Collection?

Data collection is the **first and most critical step** in any machine learning project. The quality of your data directly determines the quality of your model. As the saying goes:

> **"Garbage In, Garbage Out"**

### Why is it Important?

1. **Foundation for Learning**: ML models learn patterns from data
2. **Determines Model Performance**: Better data → Better predictions
3. **Reveals Problem Complexity**: Understanding data helps choose right algorithms
4. **Identifies Limitations**: Shows what's possible and what's not

### The Data Collection Process

```
Real World → Data Generation → Data Storage → Data Loading → Data Understanding
```

In our project:
- **Real World**: Student academic performance
- **Data Generation**: Surveys/records from Portuguese schools
- **Data Storage**: CSV file format
- **Data Loading**: Reading into pandas DataFrame
- **Data Understanding**: This phase!

---

## 2. Understanding Our Dataset {#2-understanding-our-dataset}

### Dataset Context

**Domain**: Education / Academic Performance  
**Problem Type**: Regression (predicting continuous values)  
**Source**: Portuguese secondary schools  
**Subject**: Mathematics course  
**Students**: ~395 students  
**Time Period**: Academic year data

### Why Student Performance Prediction?

1. **Early Intervention**: Identify at-risk students early
2. **Resource Allocation**: Target support where needed
3. **Understanding Factors**: What influences academic success?
4. **Policy Making**: Data-driven educational decisions

### Our Specific Dataset: Student Math Performance

We have **33 attributes** divided into categories:

1. **Demographic** (4): school, sex, age, address
2. **Family Background** (10): family size, parent status, education, jobs, etc.
3. **School-Related** (10): study time, failures, support, activities, etc.
4. **Social/Lifestyle** (6): going out, alcohol, relationships, free time
5. **Health** (1): health status
6. **Academic Performance** (3): G1, G2, G3 (our target)

---

## 3. Feature Types and Their Importance {#3-feature-types}

### 3.1 Types of Variables

#### A. Based on Data Type

**1. Numerical Variables**
- Values are numbers with mathematical meaning
- Can perform arithmetic operations

**Continuous**: Can take any value in a range
- Examples in our data: `age` (15-22), `absences` (0-93)
- Infinite possible values within range

**Discrete**: Only specific integer values
- Examples: `failures` (0,1,2,3,4), `Medu` (0,1,2,3,4)
- Countable number of values

**2. Categorical Variables**
- Represent categories or groups
- Cannot perform meaningful arithmetic

**Nominal**: No inherent order
- Examples: `school` (GP/MS), `Mjob` (teacher/health/services/at_home/other)
- Categories are just labels

**Ordinal**: Have meaningful order
- Examples: `Medu` (0<1<2<3<4), `studytime` (1<2<3<4)
- Order matters, but intervals may not be equal

**Binary**: Only two categories
- Examples: `sex` (F/M), `internet` (yes/no)
- Special case of nominal

#### B. Based on Role in ML

**1. Features (Independent Variables, X)**
- Input variables used for prediction
- In our case: 32 attributes (all except G3)
- Notation: \( X = [x_1, x_2, ..., x_{32}] \)

**2. Target (Dependent Variable, y)**
- Output variable we want to predict
- In our case: G3 (final grade)
- Notation: \( y \)

### 3.2 Why Feature Types Matter

1. **Preprocessing Requirements**:
   - Numerical: May need scaling/normalization
   - Categorical: Need encoding (one-hot, label encoding)
   
2. **Model Assumptions**:
   - Linear regression assumes numerical inputs
   - Must convert categorical to numerical

3. **Interpretation**:
   - Coefficient for numerical: "1 unit increase in X → β change in y"
   - Coefficient for binary: "Difference between categories"

---

## 4. Statistical Foundations {#4-statistical-foundations}

### 4.1 Descriptive Statistics

These summarize and describe data characteristics.

#### Measures of Central Tendency

**1. Mean (Average)**
\[
\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
\]

- Sum all values, divide by count
- Sensitive to outliers
- Example: Average age of students

**2. Median**
- Middle value when sorted
- Robust to outliers
- Better for skewed distributions

**3. Mode**
- Most frequent value
- Useful for categorical data
- Example: Most common reason for school choice

#### Measures of Spread (Variability)

**1. Range**
\[
\text{Range} = \max(x) - \min(x)
\]

- Simplest measure of spread
- Very sensitive to outliers

**2. Variance**
\[
\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2
\]

- Average squared deviation from mean
- Units are squared (hard to interpret)

**3. Standard Deviation**
\[
\sigma = \sqrt{\sigma^2} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2}
\]

- Square root of variance
- Same units as original data
- Most commonly used measure of spread

**Interpretation**: 
- Small σ: Data points close to mean (less variability)
- Large σ: Data points spread out (more variability)

#### Percentiles and Quartiles

**Percentiles**: Value below which a percentage of data falls
- 25th percentile (Q1): 25% of data below this
- 50th percentile (Q2): Median
- 75th percentile (Q3): 75% of data below this

**Interquartile Range (IQR)**
\[
\text{IQR} = Q3 - Q1
\]

- Range of middle 50% of data
- Used for outlier detection

### 4.2 Distributions

A distribution shows how values are spread across the range.

#### Normal Distribution (Gaussian)

\[
f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}(\frac{x-\mu}{\sigma})^2}
\]

**Properties**:
- Bell-shaped, symmetric
- Mean = Median = Mode
- 68% within 1σ, 95% within 2σ, 99.7% within 3σ
- Many natural phenomena follow this

**Why it matters**: Linear regression assumes errors are normally distributed

#### Skewness

Measure of asymmetry in distribution.

- **Positive skew (right-skewed)**: Long tail on right, mean > median
- **Negative skew (left-skewed)**: Long tail on left, mean < median
- **Zero skew**: Symmetric (like normal distribution)

#### Kurtosis

Measure of "tailedness" - how much data in tails vs center.

- **High kurtosis**: Heavy tails, more outliers
- **Low kurtosis**: Light tails, fewer outliers

### 4.3 Data Types in Practice

For our dataset, we need to understand:

1. **What is the distribution of grades?** (G1, G2, G3)
   - Are they normally distributed?
   - Any skewness?

2. **What is the spread of study time?**
   - Do most students study similar amounts?
   - Large variability?

3. **Are there outliers in absences?**
   - Some students miss many classes?

---

## 5. Data Quality Concepts {#5-data-quality-concepts}

### 5.1 Missing Values

**Definition**: Absent or null values in dataset

**Types**:
1. **MCAR (Missing Completely At Random)**: No pattern, truly random
2. **MAR (Missing At Random)**: Related to other observed variables
3. **MNAR (Missing Not At Random)**: Related to the missing value itself

**Why they matter**: 
- Can't train models with missing values
- Removing too much data loses information
- Imputing incorrectly introduces bias

**Detection**:
```python
df.isnull().sum()  # Count missing per column
df.info()          # Shows non-null counts
```

### 5.2 Outliers

**Definition**: Data points significantly different from others

**Why they occur**:
- Measurement errors
- Data entry errors
- True extreme values
- Different population

**Detection Methods**:

**1. Statistical Method (IQR)**
\[
\text{Outlier if: } x < Q1 - 1.5 \times IQR \text{ or } x > Q3 + 1.5 \times IQR
\]

**2. Standard Deviation Method**
- Outlier if more than 3σ from mean

**3. Visual Method**
- Box plots
- Scatter plots

**Handling**:
- Remove (if errors)
- Cap (winsorization)
- Transform (log, sqrt)
- Keep (if legitimate)

### 5.3 Data Types and Consistency

**Issues to check**:
1. **Type correctness**: Is 'age' stored as number or string?
2. **Value ranges**: Are all ages between 15-22?
3. **Categorical consistency**: "Yes" vs "yes" vs "YES"
4. **Format consistency**: Dates, decimals, etc.

### 5.4 Duplicates

**Definition**: Identical rows in dataset

**Why problematic**:
- Inflates importance of duplicated samples
- Biases model training
- Overestimates performance

**Detection**:
```python
df.duplicated().sum()
```

---

## 6. Target Variable Understanding {#6-target-variable}

### 6.1 Our Target: G3 (Final Grade)

**Characteristics**:
- **Type**: Continuous numerical
- **Range**: 0 to 20 (Portuguese grading system)
- **Scale**: Different from 0-100 systems
- **Meaning**: 
  - 0-9: Fail
  - 10: Minimum pass
  - 16-20: Excellent

### 6.2 Why Understanding Target Matters

1. **Defines Problem Type**:
   - Continuous → Regression problem
   - Would be classification if predicting pass/fail

2. **Determines Metrics**:
   - Use MSE, RMSE, MAE, R² for regression
   - Not accuracy (that's for classification)

3. **Guides Model Selection**:
   - Linear regression appropriate for continuous targets
   - Need to check if relationship is linear

4. **Influences Preprocessing**:
   - May need to scale/normalize
   - Check for outliers in target too

### 6.3 Target Distribution Analysis

**Questions to answer**:
1. What's the average final grade?
2. How spread out are the grades?
3. Is distribution normal or skewed?
4. Are there many failing students?
5. Any grade inflation/deflation?

**Why it matters**:
- Skewed target may need transformation
- Imbalanced ranges affect model training
- Understanding helps interpret predictions

---

## 7. Relationships Between Features {#7-feature-relationships}

### 7.1 Correlation

**Definition**: Measure of linear relationship between two variables

**Pearson Correlation Coefficient**:
\[
r = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n}(x_i - \bar{x})^2} \sqrt{\sum_{i=1}^{n}(y_i - \bar{y})^2}}
\]

**Range**: -1 to +1

- **r = +1**: Perfect positive correlation (x↑ → y↑)
- **r = 0**: No linear correlation
- **r = -1**: Perfect negative correlation (x↑ → y↓)

**Interpretation**:
- |r| > 0.7: Strong correlation
- 0.3 < |r| < 0.7: Moderate correlation
- |r| < 0.3: Weak correlation

**Important**: Correlation ≠ Causation!

### 7.2 Feature-Target Relationships

**What to look for**:
1. Which features correlate strongly with G3?
2. Are relationships linear or non-linear?
3. Any unexpected correlations?

**Expected relationships** (hypotheses):
- **Positive correlation**: studytime, G1, G2, Medu, Fedu
- **Negative correlation**: failures, absences, goout, Dalc, Walc
- **Weak/No correlation**: sex, address (maybe)

### 7.3 Feature-Feature Relationships (Multicollinearity)

**Definition**: When features are highly correlated with each other

**Why problematic**:
- Makes it hard to determine individual feature importance
- Unstable coefficient estimates
- Difficult interpretation

**Example**: G1 and G2 likely highly correlated (both measure math ability)

**Detection**: Correlation matrix, VIF (Variance Inflation Factor)

---

## 8. Mathematical Notation and Terminology {#8-mathematical-notation}

### 8.1 Common Symbols

| Symbol | Meaning | Example |
|--------|---------|---------|
| \( n \) | Number of samples | 395 students |
| \( m \) | Number of samples (alternative) | Same as n |
| \( p \) or \( d \) | Number of features | 32 features |
| \( x_i \) | i-th sample | Student #5 |
| \( x_j \) | j-th feature | Age feature |
| \( x_i^{(j)} \) | j-th feature of i-th sample | Age of student #5 |
| \( y_i \) | Target value for i-th sample | G3 grade of student #5 |
| \( \bar{x} \) | Mean of x | Average age |
| \( \sigma \) | Standard deviation | Spread of ages |
| \( \mu \) | Population mean | True average (if we had all students) |

### 8.2 Matrix Notation

**Feature Matrix X**:
\[
X = \begin{bmatrix}
x_1^{(1)} & x_1^{(2)} & \cdots & x_1^{(32)} \\
x_2^{(1)} & x_2^{(2)} & \cdots & x_2^{(32)} \\
\vdots & \vdots & \ddots & \vdots \\
x_{395}^{(1)} & x_{395}^{(2)} & \cdots & x_{395}^{(32)}
\end{bmatrix}
\]

- Shape: (395, 32) - 395 rows (students), 32 columns (features)
- Each row: one student's data
- Each column: one feature across all students

**Target Vector y**:
\[
y = \begin{bmatrix}
y_1 \\
y_2 \\
\vdots \\
y_{395}
\end{bmatrix}
\]

- Shape: (395, 1) or (395,)
- Each element: G3 grade for one student

### 8.3 Summation Notation

\[
\sum_{i=1}^{n} x_i = x_1 + x_2 + x_3 + \cdots + x_n
\]

**Example**: Sum of all ages
\[
\sum_{i=1}^{395} \text{age}_i = \text{age}_1 + \text{age}_2 + \cdots + \text{age}_{395}
\]

---

## 9. Practical Considerations for This Project {#9-practical-considerations}

### 9.1 Dataset-Specific Considerations

**1. Grade Scale Difference**
- Portuguese system: 0-20
- Many familiar with: 0-100
- Conversion: multiply by 5 (but we won't, to preserve original meaning)

**2. Including G1 and G2**
- **Advantage**: Strong predictors of G3
- **Consideration**: Are we predicting or just interpolating?
- **Real-world use**: Useful for mid-year predictions

**3. Categorical Encoding**
- Many binary yes/no variables
- Nominal variables (Mjob, Fjob, reason)
- Will need encoding strategy

**4. Sample Size**
- ~395 students is moderate
- 32 features is substantial
- Ratio of samples to features: ~12:1 (reasonable)

### 9.2 Questions to Answer in Phase 1

1. **Data Loading**:
   - Can we successfully load the CSV?
   - What's the delimiter? (likely semicolon for European data)
   
2. **Data Shape**:
   - How many students (rows)?
   - How many attributes (columns)?
   - Matches expectations?

3. **Data Types**:
   - Are numeric columns stored as numbers?
   - Are categorical columns strings?
   - Any type conversion needed?

4. **Missing Values**:
   - Any missing data?
   - Which columns?
   - How much?

5. **Basic Statistics**:
   - What's the average final grade?
   - Age distribution?
   - Study time patterns?

6. **Initial Observations**:
   - Any obvious issues?
   - Data quality concerns?
   - Interesting patterns?

### 9.3 Tools We'll Use

**Python Libraries**:
```python
import pandas as pd          # Data manipulation
import numpy as np           # Numerical operations
import matplotlib.pyplot as plt  # Visualization
import seaborn as sns        # Statistical visualization
```

**Key Functions**:
```python
# Loading
df = pd.read_csv('file.csv', sep=';')

# Inspection
df.head()           # First 5 rows
df.info()           # Data types, non-null counts
df.describe()       # Statistical summary
df.shape            # (rows, columns)
df.columns          # Column names
df.dtypes           # Data types

# Missing values
df.isnull().sum()   # Count missing per column
df.isnull().mean()  # Percentage missing

# Unique values
df['column'].unique()        # All unique values
df['column'].value_counts()  # Frequency of each value
```

---

## 10. Phase 1 Implementation Goals {#10-implementation-goals}

### 10.1 Deliverables

By the end of Phase 1, you should have:

1. **✅ Loaded Dataset**: Successfully read CSV into pandas DataFrame
2. **✅ Data Dictionary**: Complete understanding of all 33 attributes
3. **✅ Initial Inspection Report**: Document with:
   - Dataset dimensions
   - Data types
   - Missing value analysis
   - Basic statistical summary
   - Sample records
4. **✅ Clean Dataset**: Saved to `data/raw/` (already done)

### 10.2 Implementation Steps

**Step 1**: Import necessary libraries
**Step 2**: Load the dataset (handle delimiter correctly)
**Step 3**: Display basic information (shape, columns, types)
**Step 4**: Check for missing values
**Step 5**: Generate statistical summary
**Step 6**: Examine sample records
**Step 7**: Check unique values for categorical variables
**Step 8**: Document findings

### 10.3 Success Criteria

You understand Phase 1 when you can answer:

- ✅ How many students and features do we have?
- ✅ What types of variables exist in the dataset?
- ✅ Are there any missing values?
- ✅ What's the range of the target variable (G3)?
- ✅ What's the average final grade?
- ✅ Which features are numerical vs categorical?
- ✅ Are there any immediate data quality issues?

### 10.4 Learning Objectives

After completing Phase 1, you should be able to:

1. **Explain** the importance of data collection and understanding
2. **Identify** different types of variables (numerical, categorical, ordinal, nominal)
3. **Calculate** basic descriptive statistics (mean, median, std, quartiles)
4. **Interpret** statistical summaries
5. **Detect** missing values and understand their implications
6. **Use** pandas for data inspection
7. **Document** findings clearly

### 10.5 Common Pitfalls to Avoid

1. **Rushing through inspection**: Take time to understand each column
2. **Ignoring data types**: Wrong types cause errors later
3. **Not checking missing values**: Discover issues early
4. **Skipping documentation**: Write down observations
5. **Not understanding domain**: Context matters (e.g., grade scale)

---

## Summary

Phase 1 is about **understanding before doing**. You're building a mental model of:
- What data you have
- What it represents
- What quality it is
- What patterns might exist
- What challenges you'll face

This foundation is critical for all subsequent phases. Take your time, be thorough, and ask questions about anything unclear.

---

## Next Steps

After completing Phase 1:
1. Create conclusion document summarizing findings
2. Identify any concerns or interesting observations
3. Prepare questions for Phase 2 (EDA)
4. Get ready for deeper analysis and visualization

---

**Remember**: Good data scientists spend 80% of time understanding and preparing data, only 20% on modeling. This phase is not a formality—it's essential!

---

*Study this document thoroughly before starting Phase 1 implementation.*

