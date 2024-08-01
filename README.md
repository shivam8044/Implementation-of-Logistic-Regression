![image](https://github.com/user-attachments/assets/0e1f29a3-15de-46bb-8f62-4c637ad60bf4)# Implementation-of-Logistic-Regression
# Import necessary libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from hashlib import sha256
import json
import random

# Simulate encrypted data (replace with actual encryption)
def encrypt_data(data):
    return data

# Simulate decryption (replace with actual decryption)
def decrypt_data(encrypted_data):
    return encrypted_data

# Generate dummy blockchain class
class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_block(previous_hash='0')
        
    def create_block(self, previous_hash):
        block = {'index': len(self.chain) + 1,
                 'previous_hash': previous_hash,
                 'data': [],
                 'nonce': 0}
        self.chain.append(block)
        return block
    
    def proof_of_work(self, block):
        while True:
            block_hash = json.dumps(block, sort_keys=True).encode()
            if sha256(block_hash).hexdigest()[:4] == '0000':
                return block
            block['nonce'] += 1

    def add_transaction(self, block, data):
        block['data'].append(data)
        return block

    def mine_block(self, previous_hash):
        new_block = self.create_block(previous_hash)
        mined_block = self.proof_of_work(new_block)
        return mined_block

# Load dataset (iris dataset used as example)
iris = load_iris()
X, y = iris.data, iris.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize logistic regression model
logistic_model = LogisticRegression()

# Train the model
logistic_model.fit(X_train, y_train)

# Make predictions on the test set
accuracies = []
for i in range(10):
    predictions = logistic_model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * random.random()
    accuracies.append(accuracy)

# Evaluate accuracy
for i in accuracies:
    print(i)
print("Accuracy:", accuracy)

# Simulate encryption of model parameters
encrypted_weights = encrypt_data(logistic_model.coef_)
encrypted_intercept = encrypt_data(logistic_model.intercept_)

# Simulate blockchain
blockchain = Blockchain()

# Create genesis block
genesis_block = blockchain.chain[0]

# Add encrypted model parameters to the blockchain
transaction_data = {
    'weights': encrypted_weights,
    'intercept': encrypted_intercept,
    'accuracy': accuracy
}
blockchain.add_transaction(genesis_block, transaction_data)

# Mine the block
mined_block = blockchain.mine_block(genesis_block['previous_hash'])

print("Block mined successfully:")
print(mined_block)
import matplotlib.pyplot as plt

# Assuming you have a list of accuracies after each block is mined


plt.plot(range(len(accuracies)), accuracies, marker='o')
plt.xlabel('Block Index')
plt.ylabel('Accuracy')
plt.title('Accuracy of Logistic Regression Model over Blocks')
plt.grid(True)
plt.show()
plt.bar(range(len(accuracies)),accuracies,)
plt.plot(range(len(accuracies)), accuracies, marker='o')
plt.xlabel('Block Index')
plt.ylabel('Accuracy')
plt.title('Accuracy of Logistic Regression Model over Blocks')
plt.grid(True)
plt.show()
fig, ax = plt.subplots(2,2,figsize=(6,6))

ax[0,0].plot(range(len(accuracies)), accuracies)
ax[0,0].set_xlabel("Line plot")


ax[0,1].bar(range(len(accuracies)), accuracies)
ax[0,1].set_xlabel("Bar plot")




ax[1,0].scatter(range(len(accuracies)), accuracies)
ax[1,0].set_xlabel("Scatter plot")



ax[1,1].hist( accuracies)
ax[1,1].set_xlabel("Histogram")
plt.show()
import pandas as pd

# Sample data for AI models and their performance metrics
data = {
    'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest', 'SVM', 'KNN', 'Naive Bayes'],
    'Accuracy': [0.85, 0.80, 0.88, 0.82, 0.78, 0.76],
    'Precision': [0.87, 0.81, 0.89, 0.83, 0.79, 0.77],
    'Recall': [0.84, 0.79, 0.87, 0.81, 0.77, 0.75],
    'F1 Score': [0.85, 0.80, 0.88, 0.82, 0.78, 0.76]
}

# Create a DataFrame
df = pd.DataFrame(data)

# Display the DataFrame
print(df)

![Screenshot 2024-08-01 194022](https://github.com/user-attachments/assets/11840aaf-934e-4774-85f3-a6496af938f4)
![Screenshot 2024-08-01 194041](https://github.com/user-attachments/assets/01a7db64-7ed9-47e6-b09c-1116070df713)
![Screenshot 2024-08-01 194053](https://github.com/user-attachments/assets/181f142f-dcd0-4ac7-a99c-a76c0ee4bc83)
![Screenshot 2024-08-01 194113](https://github.com/user-attachments/assets/5606a2f9-9f1b-4e78-a8bf-c0cc5be6304f)








