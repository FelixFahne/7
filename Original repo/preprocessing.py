import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt 
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

# Define your Excel file paths
excel_files = [
    '/Users/gaowei/Downloads_annotations(1).xlsx',
    '/Users/gaowei/Downloads_annotations(2).xlsx',
    '/Users/gaowei/Downloads_annotations.xlsx'
]

# Define corresponding CSV file paths
csv_files = [
    '/Users/gaowei/Downloads_annotations(1).csv',
    '/Users/gaowei/Downloads_annotations(2).csv',
    '/Users/gaowei/Downloads_annotations.csv'
]

# Loop through each Excel file, read it, and save it as a CSV file
for excel_path, csv_path in zip(excel_files, csv_files):
    # Read the Excel file
    df = pd.read_excel(excel_path)
    
    # Save the DataFrame to a CSV file
    df.to_csv(csv_path, index=False)  # Set index=False to avoid saving the DataFrame index as a separate column

print("All files have been converted to CSV format.")


def assign_tone(row):
    if row['backchannels'] > 0 or row['code-switching for communicative purposes'] > 0 or row['collaborative finishes'] > 0:
        return 'Informal'
    elif row['subordinate clauses'] > 0 or row['impersonal subject + non-factive verb + NP'] > 0:
        return 'Formal'
    else:
        return 'Neutral'  # Default to Neutral if none of the criteria match

# Apply the function to each segment
summary_label_counts_by_segment['Tone'] = summary_label_counts_by_segment.apply(assign_tone, axis=1)

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

