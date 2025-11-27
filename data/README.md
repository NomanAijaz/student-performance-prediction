# Dataset Documentation

## Overview
This directory contains the Student Performance dataset for Mathematics course.

**Source**: UCI Machine Learning Repository - Student Performance Dataset  
**File**: `student-mat.csv`  
**Records**: 395 students  
**Attributes**: 33 (30 features + 3 grade variables)  
**Target Variable**: G3 (final grade)

---

## Dataset Structure

### File Locations
- **Raw Data**: `raw/student-mat.csv` - Original, unmodified dataset
- **Processed Data**: `processed/` - Cleaned and preprocessed datasets (created during Phase 3)

---

## Attribute Descriptions

### 1. School Information
| Attribute | Type | Description | Values |
|-----------|------|-------------|--------|
| `school` | Binary | Student's school | "GP" (Gabriel Pereira) or "MS" (Mousinho da Silveira) |

### 2. Demographic Information
| Attribute | Type | Description | Values |
|-----------|------|-------------|--------|
| `sex` | Binary | Student's gender | "F" (female) or "M" (male) |
| `age` | Numeric | Student's age | 15 to 22 years |
| `address` | Binary | Home address type | "U" (urban) or "R" (rural) |

### 3. Family Background
| Attribute | Type | Description | Values |
|-----------|------|-------------|--------|
| `famsize` | Binary | Family size | "LE3" (≤3) or "GT3" (>3) |
| `Pstatus` | Binary | Parent's cohabitation status | "T" (together) or "A" (apart) |
| `Medu` | Numeric | Mother's education level | 0-4 (0=none, 1=primary, 2=5th-9th, 3=secondary, 4=higher) |
| `Fedu` | Numeric | Father's education level | 0-4 (same scale as Medu) |
| `Mjob` | Nominal | Mother's job | "teacher", "health", "services", "at_home", "other" |
| `Fjob` | Nominal | Father's job | "teacher", "health", "services", "at_home", "other" |
| `guardian` | Nominal | Student's guardian | "mother", "father", "other" |
| `famrel` | Numeric | Quality of family relationships | 1 (very bad) to 5 (excellent) |
| `famsup` | Binary | Family educational support | "yes" or "no" |

### 4. School-Related Factors
| Attribute | Type | Description | Values |
|-----------|------|-------------|--------|
| `reason` | Nominal | Reason to choose this school | "home", "reputation", "course", "other" |
| `traveltime` | Numeric | Home to school travel time | 1 (<15 min), 2 (15-30 min), 3 (30-60 min), 4 (>1 hour) |
| `studytime` | Numeric | Weekly study time | 1 (<2 hrs), 2 (2-5 hrs), 3 (5-10 hrs), 4 (>10 hrs) |
| `failures` | Numeric | Number of past class failures | 0-4 (4 means 4 or more) |
| `schoolsup` | Binary | Extra educational school support | "yes" or "no" |
| `paid` | Binary | Extra paid classes (Math) | "yes" or "no" |
| `activities` | Binary | Extra-curricular activities | "yes" or "no" |
| `nursery` | Binary | Attended nursery school | "yes" or "no" |
| `higher` | Binary | Wants to pursue higher education | "yes" or "no" |
| `absences` | Numeric | Number of school absences | 0 to 93 |

### 5. Social & Lifestyle Factors
| Attribute | Type | Description | Values |
|-----------|------|-------------|--------|
| `internet` | Binary | Internet access at home | "yes" or "no" |
| `romantic` | Binary | In a romantic relationship | "yes" or "no" |
| `freetime` | Numeric | Free time after school | 1 (very low) to 5 (very high) |
| `goout` | Numeric | Going out with friends | 1 (very low) to 5 (very high) |
| `Dalc` | Numeric | Workday alcohol consumption | 1 (very low) to 5 (very high) |
| `Walc` | Numeric | Weekend alcohol consumption | 1 (very low) to 5 (very high) |

### 6. Health
| Attribute | Type | Description | Values |
|-----------|------|-------------|--------|
| `health` | Numeric | Current health status | 1 (very bad) to 5 (very good) |

### 7. Academic Performance (Target Variables)
| Attribute | Type | Description | Values |
|-----------|------|-------------|--------|
| `G1` | Numeric | First period grade | 0 to 20 |
| `G2` | Numeric | Second period grade | 0 to 20 |
| `G3` | Numeric | **Final grade (TARGET)** | 0 to 20 |

---

## Target Variable

**G3 (Final Grade)** is our target variable for prediction.

- **Range**: 0 to 20 (Portuguese grading system)
- **Type**: Continuous numeric
- **Interpretation**: 
  - 0-9: Fail
  - 10-20: Pass (10 is minimum passing grade)
  - 16-20: Excellent performance

---

## Prediction Task

**Objective**: Predict G3 (final grade) using all 30 features plus G1 and G2 grades.

**Features Used** (32 total):
- All demographic, family, school, social, and lifestyle attributes (30)
- First period grade (G1)
- Second period grade (G2)

**Why include G1 and G2?**
- They provide strong predictive power for final grade
- Realistic scenario: predicting final outcome based on intermediate performance
- Allows us to understand progression patterns

---

## Data Quality Notes

- **Missing Values**: To be checked during Phase 1
- **Outliers**: To be identified during Phase 2 (EDA)
- **Encoding**: Categorical variables need encoding for ML models
- **Scaling**: Numeric features have different ranges, will need normalization

---

## Important Considerations

1. **Grade Scale**: Grades are 0-20, not 0-100 (Portuguese system)
2. **Binary Variables**: Encoded as strings ("yes"/"no"), need conversion to 0/1
3. **Ordinal Variables**: Some numeric variables are ordinal (e.g., Medu, studytime)
4. **Nominal Variables**: Some categorical variables have no order (e.g., Mjob, reason)

---

## References

- **Original Source**: P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. 2008.
- **Repository**: UCI Machine Learning Repository
- **Related Files**: 
  - `student-por.csv`: Portuguese language course (not used in this project)
  - `student.txt`: Additional documentation

---

*Last Updated*: Phase 0 - Initial Setup

