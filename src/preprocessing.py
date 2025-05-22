import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

import os
from pathlib import Path

WORKDIR = Path(os.getenv("SLDEA_WORKDIR", "/tmp/space"))

# Annotation CSV files are expected to reside in ``WORKDIR``
csv_files = [WORKDIR / f"annotations({i}).csv" for i in range(1, 6)]

# Read all available annotation CSVs
frames = [pd.read_csv(p) for p in csv_files if p.exists()]
if not frames:
    raise FileNotFoundError(f"No annotation CSVs found in {WORKDIR}")
df = pd.concat(frames, ignore_index=True)

# Summarize label counts per dialogue segment (or similar unit)
summary_label_counts_by_segment = (
    df.groupby("dialogue_segment").sum(numeric_only=True)
    if "dialogue_segment" in df.columns
    else pd.DataFrame()
)


def assign_tone(row):
    """Return a tone label based on available label counts."""
    if (
        row.get("backchannels", 0) > 0
        or row.get("code-switching for communicative purposes", 0) > 0
        or row.get("collaborative finishes", 0) > 0
    ):
        return "Informal"
    if (
        row.get("subordinate clauses", 0) > 0
        or row.get("impersonal subject + non-factive verb + NP", 0) > 0
    ):
        return "Formal"
    return "Neutral"

# Apply the function only if groupby was successful
if not summary_label_counts_by_segment.empty:
    summary_label_counts_by_segment["Tone"] = summary_label_counts_by_segment.apply(
        assign_tone, axis=1
    )

# Overview of tone assignments
tone_assignments = summary_label_counts_by_segment['Tone'].value_counts()

# Visualize the distribution of assigned tones across segments
plt.figure(figsize=(8, 5))
tone_assignments.plot(kind='bar')
plt.title('Distribution of Assigned Tones Across Dialogue Segments')
plt.xlabel('Tone')
plt.ylabel('Number of Segments')
plt.xticks(rotation=0)
plt.show()

# Now that we have assigned tones, we could explore the relationship between these tones and specific labels
# This step is illustrative and based on the simplified criteria for tone assignment
tone_assignments


# Assuming `df` is a DataFrame with dialogue identifiers and the constructed dialogue-level labels

# Feature Engineering: Summarize token-level labels into dialogue-level features
features = df.groupby('dialogue_id').agg({
    'token_label_type1': 'sum',
    'token_label_type2': 'sum',
    # Add more as needed
})

# Assume `dialogue_labels` is a DataFrame with our dialogue-level labels
dialogue_labels = df.groupby('dialogue_id').agg({
    'OverallToneChoice': 'first',  # Assuming a method to assign these labels
    'TopicExtension': 'first'
})

# Join features with labels
data_for_regression = features.join(dialogue_labels)

# Split data into features (X) and labels (y)
X = data_for_regression.drop(['OverallToneChoice', 'TopicExtension'], axis=1)
y = data_for_regression[['OverallToneChoice', 'TopicExtension']]

# Regression analysis (simplified)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model for 'Overall Tone Choice'
model_tone = LinearRegression().fit(X_train, y_train['OverallToneChoice'])

# Predict and evaluate 'Overall Tone Choice'
# (Evaluation steps would go here)

# Repeat for 'Topic Extension'
model_topic = LinearRegression().fit(X_train, y_train['TopicExtension'])

# Predict and evaluate 'Topic Extension'
# (Evaluation steps would go here)





# Assuming `df` is a DataFrame with dialogue identifiers and the constructed dialogue-level labels

# Feature Engineering: Summarize token-level labels into dialogue-level features
features = df.groupby('dialogue_id').agg({
    'token_label_type1': 'sum',
    'token_label_type2': 'sum',
    # Add more as needed
})

# Assume `dialogue_labels` is a DataFrame with our dialogue-level labels
dialogue_labels = df.groupby('dialogue_id').agg({
    'OverallToneChoice': 'first',  # Assuming a method to assign these labels
    'TopicExtension': 'first'
})

# Join features with labels
data_for_regression = features.join(dialogue_labels)

# Split data into features (X) and labels (y)
X = data_for_regression.drop(['OverallToneChoice', 'TopicExtension'], axis=1)
y = data_for_regression[['OverallToneChoice', 'TopicExtension']]

# Regression analysis (simplified)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear regression model for 'Overall Tone Choice'
model_tone = LinearRegression().fit(X_train, y_train['OverallToneChoice'])

# Predict and evaluate 'Overall Tone Choice'
# (Evaluation steps would go here)

# Repeat for 'Topic Extension'
model_topic = LinearRegression().fit(X_train, y_train['TopicExtension'])

# Predict and evaluate 'Topic Extension'

# Persist the merged annotations so other scripts can access them
df.to_csv(WORKDIR / "feature_label.csv", index=False)
