import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DecisionTree:

    '''
    Decision Tree is a tree based algorithm providing a multi-level conditional 
    architecture to make predictions by evaluating the incremental information gain.
    '''

    def __init__(self, max_depth=4, depth=1):
        self.left = None
        self.right = None
        self.max_depth = max_depth
        self.depth = depth

    def fit(self, data, target):
        self.data = data
        self.target = target
        self.independent = self.data.columns.tolist()
        self.independent.remove(target)

    def __calculate_impurity_score(self, data):
        if data is None or data.empty:
            return 0
        p_i, _ = data.value_counts().apply(lambda x: x/len(data)).tolist()
        return p_i * (1 - p_i) * 2

    def __find_best_split_for_column(self, col):
        x = self.data[col]
        unique_values = x.unique()
        if len(unique_values) == 1:
            return None, None
        information_gains = None
        split = None

        for val in unique_values:
            left = x <= val
            right = x > val
            left_data = self.data[left]
            right_data = self.data[right]
            left_impurity = self.__calculate_impurity_score(self.data[left][self.target])
            right_impurity = self.__calculate_impurity_score(self.data[right][self.target])
            score = self.__calculate_information_gain(len(left_data), left_impurity, len(right_data), right_impurity)

            if information_gains is None or score > information_gains:
                information_gains = score
                split = val
        
        return information_gains, split

    def __calculate_information_gain(self, left_count, left_impurity, right_count, right_impurity):
        return self.impurity_score - ((left_count/len(self.data)) * left_impurity +
                                       (right_count/len(self.data)) * right_impurity)


    def __find_best_split(self):
        
        best_split = {}
        for col in self.independent:
            information_gain, split = self.__find_best_split_for_column(col)
            if information_gain is None:
                continue
            if not best_split or information_gain > best_split["information_gain"]:
                best_split = {
                    "col": col,
                    "split": split,
                    "information_gain": information_gain
                }
        
        return best_split["split"], best_split["col"]

    def __create_branches(self):

        self.left = DecisionTree(max_depth=self.max_depth, depth=self.depth+1)
        self.right = DecisionTree(max_depth=self.max_depth, depth=self.depth+1)

        left_rows = self.data[self.data[self.split_feature] <= self.criteria]
        right_rows = self.data[self.data[self.split_feature] > self.criteria]
        self.left.fit(left_rows, self.target)
        self.right.fit(right_rows, self.target)

    @property
    def is_leaf_node(self):
        return self.left is None and self.right is None
    
    @property
    def probability(self):
        return self.data[self.target].value_counts().apply(lambda x: x/len(self.data)).tolist()
    
    def predict(self, data):
        return np.array([self.__flow_data_thru_tree(row) for _, row in data.iterrows()])
    
    def __flow_data_thru_tree(self, row):
        if self.is_leaf_node: return self.probability
        tree = self.left if row[self.split_feature] <= self.criteria else self.right
        return tree.__flow_data_thru_tree(row)


def read_csv(file_path):
    return pd.read_csv(file_path)


# Load the dataset
train_data = pd.read_csv('data/train.csv') 
test_data = pd.read_csv('data/test.csv')

# Preprocessing
# Encode categorical columns
label_encoder = LabelEncoder()
for col in ['Sex', 'Embarked']:
    if col in train_data.columns:
        train_data[col] = train_data[col].fillna('Unknown')  # Handle missing values
        train_data[col] = label_encoder.fit_transform(train_data[col])
    if col in test_data.columns:
        test_data[col] = test_data[col].fillna('Unknown')  
        test_data[col] = label_encoder.transform(test_data[col])

# Fill missing numerical values with median
for col in ['Age', 'Fare']:
    if col in train_data.columns:
        train_data[col] = train_data[col].fillna(train_data[col].median())
    if col in test_data.columns:
        test_data[col] = test_data[col].fillna(test_data[col].median())

# Split the training data
X_train, X_val, y_train, y_val = train_test_split(
    train_data.drop('Survived', axis=1), 
    train_data['Survived'], 
    test_size=0.2, 
    random_state=42
)

# Combine the training data for simplicity
train_data = pd.concat([X_train, y_train], axis=1)

# Train the decision tree
tree = DecisionTree(max_depth=5)
tree.fit(train_data, target='Survived')

# Predict probabilities on the test dataset
predicted_probabilities = tree.predict(test_data)
yes_prob = predicted_probabilities[1] if len(predicted_probabilities) > 1 else 0
no_prob = predicted_probabilities[0] 

# Output predictions
test_data['Survival_Probability'] = predicted_probabilities.tolist()
print(test_data[['PassengerId', 'Survival_Probability']])
