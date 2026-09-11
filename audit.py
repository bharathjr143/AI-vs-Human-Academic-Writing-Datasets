# ============================================================
# AI WRITING DETECTION
# TF-IDF + Logistic Regression
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

FILE_PATH = "ai_writing_detection_dataset.csv"

data = pd.read_csv(FILE_PATH)

print("\n==========================================")
print("       AI WRITING DETECTION SYSTEM")
print("==========================================")

print("\nDataset loaded successfully!")
print("Number of rows    :", data.shape[0])
print("Number of columns :", data.shape[1])

print("\nAvailable columns:")
for column in data.columns:
    print("-", column)


# ============================================================
# 2. AUTOMATICALLY FIND TEXT COLUMN
# ============================================================

text_keywords = [
    "text",
    "essay",
    "writing",
    "content",
    "article",
    "document",
    "paragraph",
    "response",
    "answer",
    "statement",
    "prompt"
]

text_column = None

# First search by column name
for column in data.columns:

    column_name = str(column).lower().strip()

    for keyword in text_keywords:

        if keyword in column_name:
            text_column = column
            break

    if text_column is not None:
        break


# If no suitable name is found, select the column
# containing the largest amount of textual information.

if text_column is None:

    object_columns = data.select_dtypes(
        include=["object"]
    ).columns

    if len(object_columns) == 0:
        raise ValueError(
            "No text column was found in the dataset."
        )

    average_lengths = {}

    for column in object_columns:
        average_lengths[column] = (
            data[column]
            .fillna("")
            .astype(str)
            .str.len()
            .mean()
        )

    text_column = max(
        average_lengths,
        key=average_lengths.get
    )


print("\nSelected TEXT column:")
print(text_column)


# ============================================================
# 3. AUTOMATICALLY FIND TARGET / LABEL COLUMN
# ============================================================

target_keywords = [
    "label",
    "target",
    "class",
    "category",
    "generated",
    "ai",
    "human",
    "source",
    "author",
    "type",
    "origin"
]

target_column = None


# Search columns by common target names

for column in data.columns:

    if column == text_column:
        continue

    column_name = str(column).lower().strip()

    for keyword in target_keywords:

        if keyword in column_name:
            target_column = column
            break

    if target_column is not None:
        break


# If target was not found by name,
# look for a column containing exactly two classes.

if target_column is None:

    for column in data.columns:

        if column == text_column:
            continue

        number_of_classes = data[column].nunique(
            dropna=True
        )

        if number_of_classes == 2:
            target_column = column
            break


if target_column is None:

    raise ValueError(
        "Could not find the AI/Human target column."
    )


print("\nSelected TARGET column:")
print(target_column)


# ============================================================
# 4. REMOVE MISSING VALUES
# ============================================================

data = data[
    [text_column, target_column]
].copy()

data[text_column] = (
    data[text_column]
    .fillna("")
    .astype(str)
    .str.strip()
)

data = data[
    data[text_column] != ""
]

data = data.dropna(
    subset=[target_column]
)

print("\nRows after cleaning:", len(data))


# ============================================================
# 5. SHOW TARGET VALUES
# ============================================================

print("\nTarget values found:")

print(
    data[target_column].value_counts()
)


# ============================================================
# 6. CONVERT AI / HUMAN LABELS TO 0 / 1
# ============================================================

target_values = list(
    data[target_column].unique()
)

print("\nUnique target values:")
print(target_values)


def convert_target(value):

    value_string = str(value).lower().strip()

    # HUMAN
    if value_string in [
        "human",
        "human-written",
        "human_written",
        "real",
        "person",
        "0",
        "false"
    ]:
        return 0

    # AI
    if value_string in [
        "ai",
        "ai-generated",
        "ai_generated",
        "generated",
        "machine",
        "machine-generated",
        "artificial",
        "1",
        "true"
    ]:
        return 1

    return np.nan


# Try automatic conversion

converted_target = data[
    target_column
].apply(convert_target)


# ============================================================
# 7. IF AUTOMATIC CONVERSION FAILS
#    USE BINARY CLASS ENCODING
# ============================================================

if converted_target.isna().any():

    unique_values = list(
        data[target_column].unique()
    )

    if len(unique_values) != 2:

        raise ValueError(
            "The target column must contain exactly "
            "two classes for AI/Human detection."
        )

    print(
        "\nCustom target labels detected."
    )

    print(
        "Mapping labels automatically:"
    )

    print(
        unique_values[0],
        "-> 0"
    )

    print(
        unique_values[1],
        "-> 1"
    )

    label_mapping = {
        unique_values[0]: 0,
        unique_values[1]: 1
    }

    converted_target = (
        data[target_column]
        .map(label_mapping)
    )


data["AI_Label"] = converted_target.astype(int)


# ============================================================
# 8. CREATE X AND Y
# ============================================================

X = data[text_column]

y = data["AI_Label"]


print("\nFinal class distribution:")

print(
    y.value_counts()
)


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\n==========================================")
print("          DATASET SPLIT")
print("==========================================")

print(
    "Training samples:",
    len(X_train)
)

print(
    "Testing samples :",
    len(X_test)
)


# ============================================================
# 10. TF-IDF FEATURE EXTRACTION
# ============================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(

    lowercase=True,

    stop_words="english",

    ngram_range=(1, 2),

    min_df=1,

    max_features=10000
)


X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)


print(
    "Number of TF-IDF features:",
    X_train_tfidf.shape[1]
)


# ============================================================
# 11. TRAIN LOGISTIC REGRESSION MODEL
# ============================================================

print("\nTraining Logistic Regression model...")

model = LogisticRegression(

    max_iter=2000,

    class_weight="balanced",

    random_state=42
)


model.fit(
    X_train_tfidf,
    y_train
)


print("Model training completed!")


# ============================================================
# 12. PREDICTION
# ============================================================

y_pred = model.predict(
    X_test_tfidf
)


# ============================================================
# 13. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n==========================================")
print("          MODEL PERFORMANCE")
print("==========================================")

print(
    "Accuracy  :",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Precision :",
    round(precision * 100, 2),
    "%"
)

print(
    "Recall    :",
    round(recall * 100, 2),
    "%"
)

print(
    "F1 Score  :",
    round(f1 * 100, 2),
    "%"
)


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

print("\n==========================================")
print("        CLASSIFICATION REPORT")
print("==========================================")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Human",
            "AI"
        ],
        zero_division=0
    )
)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

print("\n==========================================")
print("           CONFUSION MATRIX")
print("==========================================")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ============================================================
# 16. AI WRITING PREDICTION FUNCTION
# ============================================================

def detect_ai_writing(text):

    if not isinstance(text, str):

        raise ValueError(
            "Input must be text."
        )

    if text.strip() == "":

        raise ValueError(
            "Input text cannot be empty."
        )


    # Convert new text into TF-IDF

    text_features = vectorizer.transform(
        [text]
    )


    # Prediction

    prediction = model.predict(
        text_features
    )[0]


    # Probability

    probability = model.predict_proba(
        text_features
    )[0]


    if prediction == 1:

        result = "AI-Generated"

        confidence = probability[1]

    else:

        result = "Human-Written"

        confidence = probability[0]


    return result, confidence


# ============================================================
# 17. TEST NEW TEXT
# ============================================================

print("\n==========================================")
print("       TEST NEW WRITING SAMPLE")
print("==========================================")

new_text = input(
    "\nEnter a paragraph to analyze:\n"
)


result, confidence = detect_ai_writing(
    new_text
)


print("\nPrediction:")
print(result)

print(
    "Confidence:",
    round(confidence * 100, 2),
    "%"
)


# ============================================================
# 18. FINISHED
# ============================================================

print("\n==========================================")
print("        PROGRAM COMPLETED")
print("==========================================")

