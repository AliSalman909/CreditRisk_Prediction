#librareis imported 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression  #for mdoel training 
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay  #for evaluation 

# Load the training dataset
train_data = pd.read_csv(r"C:\Users\Ali's HP\Desktop\Internship\task2\train.csv")
print("Dataset Shape:", train_data.shape)
print("\nFirst 5 Rows:\n", train_data.head())
print("\nColumn Info:\n", train_data.info())
print("\nSummary Statistics:\n", train_data.describe())

# Check for missing values
print("\nMissing Values:\n", train_data.isnull().sum())

# Input missing values
# Categorical columns and use most frequent value (mode)
for column in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
    train_data[column] = train_data[column].fillna(train_data[column].mode()[0])

# Numerical columns so use median and mode 
train_data['LoanAmount'] = train_data['LoanAmount'].fillna(train_data['LoanAmount'].median())  #median for LoanAmount
train_data['Loan_Amount_Term'] = train_data['Loan_Amount_Term'].fillna(train_data['Loan_Amount_Term'].mode()[0]) # mode for Loan_Amount_Term

# Verify no missing values remain
print("\nMissing Values After Imputation:\n", train_data.isnull().sum())

# Visualization by creating histogram and other plots
plt.figure(figsize=(8, 5))
sns.histplot(train_data['LoanAmount'], kde=True)
plt.title('Distribution of Loan Amount')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.show()
#
plt.figure(figsize=(8, 5))
sns.histplot(train_data['ApplicantIncome'], kde=True)
plt.title('Distribution of Applicant Income')
plt.xlabel('Applicant Income')
plt.ylabel('Frequency')
plt.show()
#creating count plot
plt.figure(figsize=(8, 5))
sns.countplot(x='Education', hue='Loan_Status', data=train_data)
plt.title('Education vs Loan Status')
plt.xlabel('Education')
plt.ylabel('Count')
plt.show()
#create box plot
plt.figure(figsize=(8, 5))
sns.boxplot(y='LoanAmount', data=train_data)
plt.title('Box Plot of Loan Amount')
plt.ylabel('Loan Amount')
plt.show()

# Convert categorical columns to numeric using mapping (encoding)
train_data['Gender'] = train_data['Gender'].map({'Male': 1, 'Female': 0})
train_data['Married'] = train_data['Married'].map({'Yes': 1, 'No': 0})
train_data['Education'] = train_data['Education'].map({'Graduate': 1, 'Not Graduate': 0})
train_data['Self_Employed'] = train_data['Self_Employed'].map({'Yes': 1, 'No': 0})
train_data['Loan_Status'] = train_data['Loan_Status'].map({'Y': 1, 'N': 0})
train_data = pd.get_dummies(train_data, columns=['Dependents', 'Property_Area'], drop_first=True)

#drop as its an identifier and not useful in model
train_data.drop('Loan_ID', axis=1, inplace=True)

X = train_data.drop('Loan_Status', axis=1)
y = train_data['Loan_Status']

# Split the data into 80% training and 20% validation (test) sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=50)

#print the shape 
print("\n\nTraining Set Shape:", X_train.shape)
print("Validation Set Shape:", X_val.shape)

scaler = StandardScaler()
# Fit scaler on training data and transform both train and validation sets
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train Logistic Regression
model = LogisticRegression(random_state=50)
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

# Evaluate model
accuracy = accuracy_score(y_val, y_pred)
print("\n\nLogistic Regression Validation Accuracy:", accuracy)

# Create a confusion matrix for more detailed evaluation
cm = confusion_matrix(y_val, y_pred)

# Visualize the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Approved', 'Approved'])
disp.plot(cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.show()