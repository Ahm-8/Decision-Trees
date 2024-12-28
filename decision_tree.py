import numpy as np
import pandas as pd

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
        self.independent.remove(self.target)

    def predict(self, data):
        return np.array([self.__flow_data_thru_tree(row) for row in data.values])
    
    def __flow_data_thru_tree(self, row):
        return self.data[self.target].value_counts().apply(lambda x: x/len(self.data)).tolist()

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

def read_csv(file_path):
    return pd.read_csv(file_path)

if __name__ == "__main__":
    
    train = read_csv("data/train.csv")
    test = read_csv("data/test.csv")

    model = DecisionTree()
    model.fit(train, "Survived")
    predictions = __find_best_split(model)
    print(predictions)