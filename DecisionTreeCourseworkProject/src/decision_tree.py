import numpy as np

# Define a decision tree class
class DecisionTree:
    def __init__(self, train_x, train_y, test_x):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x

    def fit(self, max_depth, threshold):
        """
        finds the optimal decision tree for the training data

        param:
            int:max_depth
            float:threshold
        return:
            dict: nodes_dict
        """

        x = self.train_x
        y = self.train_y
        nodes_dict = {}
        # nodes_dict has elements in the following form:
        # 'n_{m}': {'attribute': float(:feature index),
        #           'value': float,
        #           'left': str(:the number of left node),
        #           'right': str(:the number of right node),
        #           'leaf': tuple:(Bool, label type: estimated label)}
        subset_list = [(x, y, '0')] # the tuple contains samples, labels and the subset corresponding node number
        i = 0

        while i < len(subset_list):
            # let current node be leaf node if all labels in the subset are the same or reaching max depth
            current_depth = len(subset_list[i][2]) - 1
            if np.unique(subset_list[i][1]).size == 1 or current_depth >= max_depth:
                # find the label for majority class
                unique, counts = np.unique(subset_list[i][1], return_counts=True)
                majority_label = unique[np.argmax(counts)]
                # let current node be a leaf node
                current_leaf = (True, majority_label)
                nodes_dict['n_'+subset_list[i][2]] = {'attribute': None,
                                                      'value': None,
                                                      'left': None,
                                                      'right': None,
                                                      'leaf': current_leaf}
                i += 1
                continue

            current_attribute, current_value, max_IG = self.find_best_split(subset_list[i][0], subset_list[i][1])
            if max_IG < threshold:
                # find the label for majority class
                unique, counts = np.unique(subset_list[i][1], return_counts=True)
                majority_label = unique[np.argmax(counts)]
                # let current node be a leaf node
                current_leaf = (True, majority_label)
                nodes_dict['n_' + subset_list[i][2]] = {'attribute': None,
                                                        'value': None,
                                                        'left': None,
                                                        'right': None,
                                                        'leaf': current_leaf}
                i += 1
                continue

            current_node_number = subset_list[i][2]
            current_left = current_node_number + '0'
            current_right = current_node_number + '1'
            current_leaf = (False, None)

            nodes_dict['n_' + subset_list[i][2]] = {'attribute': current_attribute,
                                                    'value': current_value,
                                                    'left': 'n_' + current_left,
                                                    'right': 'n_' + current_right,
                                                    'leaf': current_leaf}

            current_set_x = subset_list[i][0]
            current_set_y = subset_list[i][1]

            left_set = (current_set_x[current_set_x[:, current_attribute] < current_value],
                        current_set_y[current_set_x[:, current_attribute] < current_value],
                        current_left)

            right_set = (current_set_x[current_set_x[:, current_attribute] >= current_value],
                         current_set_y[current_set_x[:, current_attribute] >= current_value],
                         current_right)

            subset_list.append(left_set)
            subset_list.append(right_set)

            i += 1

        return nodes_dict



    def predict(self, nodes_dict):
        """
        predict the labels for the test data

        param:
            dict: nodes_dict
        return:
            ndarray: y
        """

        x = self.test_x
        y = np.empty(x.shape[0], dtype=self.train_y.dtype) # to store predictions
        for j in range(len(x)):
            x_j = x[j, :]
            current_node = 'n_0'
            while not nodes_dict[current_node]['leaf'][0]:
                # loop until finding the leaf node
                i = nodes_dict[current_node]['attribute']
                if x_j[i] < nodes_dict[current_node]['value']:
                    current_node = nodes_dict[current_node]['left']
                elif x_j[i] >= nodes_dict[current_node]['value']:
                    current_node = nodes_dict[current_node]['right']
            y[j] = nodes_dict[current_node]['leaf'][1]
            # fill in the jth element of y with leaf label

        return y



    def calculate_entropy(self, y):
        """
        For given set of y, calculates the entropy of the set y
        entropy = -sum(p * log2(p))

        args:
            numpy array:y
        return:
            float:entropy
        """

        unique, counts = np.unique(y, return_counts=True)
        N = len(y)
        p = counts / N # the estimated probability for each label
        entropy = -np.sum(p * np.log2(p))

        return entropy



    def find_best_split(self, x, y):
        """
        For given set of x and y, finds the best split (with maximum information gain)

        param:
            numpy array:x
            numpy array:y
        return:
            label type:attribute, float:value, float:maximum information gain
        """
        entropy_0 = self.calculate_entropy(y)
        candidates = []  # list of tuples (feature_index, split_value, IG)

        for i in range(x.shape[1]):
            # sorts samples by feature i
            sorted_idx = np.argsort(x[:, i])
            sorted_x = x[sorted_idx]
            sorted_y = y[sorted_idx]

            for j in range(len(sorted_y) - 1):
                if sorted_y[j] != sorted_y[j + 1]:
                    # candidate split between j and j+1
                    val = (sorted_x[j, i] + sorted_x[j + 1, i]) / 2
                    # partitions for this split
                    s1 = sorted_y[sorted_x[:, i] < val]
                    s2 = sorted_y[sorted_x[:, i] >= val]
                    if len(s1) == 0 or len(s2) == 0:
                        continue
                    p_left = (j + 1) / len(sorted_y)
                    p_right = 1 - p_left
                    IG = entropy_0 - p_left * self.calculate_entropy(s1) - p_right * self.calculate_entropy(s2)
                    candidates.append((int(i), float(val), float(IG)))

        if not candidates:
            return 0, 0.0, 0.0

        # pick candidate with max IG
        best = max(candidates, key=lambda t: t[2])
        return int(best[0]), best[1], best[2]

