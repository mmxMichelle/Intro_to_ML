
# --- Prediction and Evaluation Functions ---

def predict_single(x, tree):
    """
    Predicts the class label for a single sample using a trained decision tree.
    """
    if 'leaf_value' in tree:
        return tree['leaf_value']
    
    feature_index = tree['feature_index']
    threshold = tree['threshold']
    
    if x[feature_index] <= threshold:
        return predict_single(x, tree['left'])
    else:
        return predict_single(x, tree['right'])

def evaluate(test_db, trained_tree):
    """
    Evaluates the accuracy of a trained tree on a test dataset.
    """
    X_test, y_test = test_db[:, :-1], test_db[:, -1]
    y_pred = [predict_single(x, trained_tree) for x in X_test]
    accuracy = np.mean(y_pred == y_test)
    return accuracy

def get_predictions(test_db, trained_tree):
    """Gets predictions for a whole test set."""
    X_test = test_db[:, :-1]
    return np.array([predict_single(x, trained_tree) for x in X_test])

def confusion_matrix(y_true, y_pred):
    """Computes the confusion matrix."""
    num_classes = len(np.unique(np.concatenate((y_true, y_pred))))
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[int(true-1), int(pred-1)] += 1
    return cm

def calculate_metrics(cm):
    """Calculates accuracy, precision, recall, and F1-score from a confusion matrix."""
    num_classes = cm.shape[0]
    accuracy = np.trace(cm) / np.sum(cm)
    
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        
    return accuracy, precision, recall, f1

def cross_validation(dataset, k=10):
    """
    Performs k-fold cross-validation.
    """
    np.random.shuffle(dataset)
    folds = np.array_split(dataset, k)
    
    all_y_true = []
    all_y_pred = []
    tree_depths = []

    for i in range(k):
        test_set = folds[i]
        train_set_list = [folds[j] for j in range(k) if i != j]
        train_set = np.concatenate(train_set_list)
        tree, depth = decision_tree_learning(train_set)
        tree_depths.append(depth)
        y_true = test_set[:, -1]
        y_pred = get_predictions(test_set, tree)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
    return np.array(all_y_true), np.array(all_y_pred), np.mean(tree_depths)

# --- Pruning Functions ---

def count_nodes(tree):
    """Recursively counts nodes in the tree."""
    if 'leaf_value' in tree:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])
    
def get_prunable_nodes(tree, parent=None, direction=None):
    """Finds nodes that are connected to two leaves."""
    prunable = []
    if 'leaf_value' not in tree:
        is_left_leaf = 'leaf_value' in tree['left']
        is_right_leaf = 'leaf_value' in tree['right']
        
        if is_left_leaf and is_right_leaf:
            prunable.append({'node': tree, 'parent': parent, 'direction': direction})
        else:
            prunable.extend(get_prunable_nodes(tree['left'], parent=tree, direction='left'))
            prunable.extend(get_prunable_nodes(tree['right'], parent=tree, direction='right'))
    return prunable

def prune_tree(tree, validation_set):
    """
    Prunes a decision tree based on validation error.
    """
    import copy
    
    pruned_tree = copy.deepcopy(tree)
    
    while True:
        initial_accuracy = evaluate(validation_set, pruned_tree)
        prunable_nodes = get_prunable_nodes(pruned_tree)
        
        best_pruned_tree = None
        best_accuracy = initial_accuracy
        
        node_to_prune_info = None

        for node_info in prunable_nodes:
            temp_tree = copy.deepcopy(pruned_tree)
            current_node_info = next((item for item in get_prunable_nodes(temp_tree) if item['node'] == node_info['node']), None)
            if current_node_info:
                parent = current_node_info['parent']
                direction = current_node_info['direction']
                y_left = get_all_labels(current_node_info['node']['left'])
                y_right = get_all_labels(current_node_info['node']['right'])
                all_y = np.concatenate((y_left, y_right))
                if len(all_y) == 0:
                    continue
                majority_class = Counter(all_y).most_common(1)[0][0]
                leaf_node = {'leaf_value': majority_class}
                if parent:
                    parent[direction] = leaf_node
                else:
                    temp_tree = leaf_node
                current_accuracy = evaluate(validation_set, temp_tree)
                if current_accuracy >= best_accuracy:
                    best_accuracy = current_accuracy
                    best_pruned_tree = temp_tree
                    node_to_prune_info = node_info
        
        if best_pruned_tree:
            pruned_tree = best_pruned_tree
        else:
            break
            
    return pruned_tree

def get_all_labels(tree):
    """Helper to get all labels from a subtree for pruning."""
    if 'leaf_value' in tree:
        return np.array([tree['leaf_value']])
    else:
        left_labels = get_all_labels(tree['left'])
        right_labels = get_all_labels(tree['right'])
        return np.concatenate((left_labels, right_labels))

def nested_cross_validation(dataset, k=10):
    """
    Performs nested k-fold cross-validation for pruning.
    """
    np.random.shuffle(dataset)
    folds = np.array_split(dataset, k)
    
    all_y_true_pruned = []
    all_y_pred_pruned = []
    pruned_tree_depths = []

    for i in range(k):
        test_set = folds[i]
        train_val_set = np.concatenate([folds[j] for j in range(k) if i != j])
        split_idx = int(0.9 * len(train_val_set))
        train_set = train_val_set[:split_idx]
        validation_set = train_val_set[split_idx:]
        tree, _ = decision_tree_learning(train_set)
        pruned_tree = prune_tree(tree, validation_set)
        pruned_tree_depths.append(get_tree_depth(pruned_tree))
        y_true = test_set[:, -1]
        y_pred = get_predictions(test_set, pruned_tree)
        all_y_true_pruned.extend(y_true)
        all_y_pred_pruned.extend(y_pred)

    return np.array(all_y_true_pruned), np.array(all_y_pred_pruned), np.mean(pruned_tree_depths)

def get_tree_depth(tree):
    """Calculates the depth of a given tree."""
    if 'leaf_value' in tree:
        return 0
    return 1 + max(get_tree_depth(tree['left']), get_tree_depth(tree['right']))

# --- Visualization Functions (Bonus) ---

decision_node_style = dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="black")
leaf_node_style = dict(boxstyle="circle", fc="honeydew", ec="black")
arrow_args = dict(arrowstyle="<-", color="black", shrinkA=0, shrinkB=10)

def plot_node(ax, node_text, center_pt, parent_pt, node_style, node_fontsize=12):  # [修改] 新增 node_fontsize 形参
    """Plots a single node and the arrow connecting it to its parent."""
    ax.annotate(
        node_text, xy=parent_pt, xycoords='axes fraction',
        xytext=center_pt, textcoords='axes fraction',
        va="center", ha="center", bbox=node_style, arrowprops=arrow_args,
        fontsize=node_fontsize  # [修改] 使用可调字号
    )

def get_num_leafs(tree):
    """Counts the number of leaves in a tree."""
    if 'leaf_value' in tree:
        return 1
    return get_num_leafs(tree['left']) + get_num_leafs(tree['right'])

def plot_tree_recursive(ax, tree, parent_pt, x_offset, y_offset, total_width, total_depth,
                        node_fontsize=12):  # [修改] 传递字号
    """
    Recursively plots the decision tree with improved spacing and aesthetics.
    """
    num_leaves = get_num_leafs(tree)
    center_pt = (x_offset + (0.5 * num_leaves) / total_width, y_offset)

    if 'leaf_value' in tree:
        node_text = f"Room:\n{int(tree['leaf_value'])}"
        plot_node(ax, node_text, center_pt, parent_pt, leaf_node_style, node_fontsize)  # [修改]
        return

    node_text = f"X[{tree['feature_index']}] <= {tree['threshold']:.2f}"
    plot_node(ax, node_text, center_pt, parent_pt, decision_node_style, node_fontsize)  # [修改]

    child_y_offset = y_offset - 1.0 / total_depth
    plot_tree_recursive(ax, tree['left'], center_pt, x_offset, child_y_offset,
                        total_width, total_depth, node_fontsize)  # [修改]
    right_x_offset = x_offset + float(get_num_leafs(tree['left'])) / total_width
    plot_tree_recursive(ax, tree['right'], center_pt, right_x_offset, child_y_offset,
                        total_width, total_depth, node_fontsize)  # [修改]

def plot_tree(tree, title="Decision Tree", title_pad=40, node_fontsize=12):  # [修改] 新增 title_pad 与 node_fontsize
    """
    Creates a plot of the decision tree. This is the main plotting function.
    """
    fig = plt.figure(figsize=(24, 14), facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    ax = plt.subplot(111, frameon=False, **axprops)
    
    total_width = float(get_num_leafs(tree))
    total_depth = float(get_tree_depth(tree)) + 1
    
    plot_tree_recursive(ax, tree, (0.5, 1.0), 0.0, 1.0, total_width, total_depth,
                        node_fontsize=node_fontsize)  # [修改]

    plt.title(title, fontsize=20, pad=title_pad)  # [修改] 通过 pad 拉开标题与树的距离
    plt.savefig(f"{title.replace(' ', '_')}.png", bbox_inches='tight', dpi=150)
    print(f"A clearer tree visualization saved as '{title.replace(' ', '_')}.png'")
    plt.show()

# --- Main Execution ---

def main():
    """
    Main function to run the coursework experiments.
    """
    clean_data_path = '/Users/mamingxuan/Downloads/For 70050/wifi_db/clean_dataset.txt'
    noisy_data_path = '/Users/mamingxuan/Downloads/For 70050/wifi_db/noisy_dataset.txt'

    if not os.path.exists(clean_data_path) or not os.path.exists(noisy_data_path):
        print("Error: Dataset files not found.")
        print("Please make sure 'clean_dataset.txt' and 'noisy_dataset.txt' are inside a 'WIFI_db' folder.")
        return

    clean_dataset = np.loadtxt(clean_data_path)
    noisy_dataset = np.loadtxt(noisy_data_path)
    
    print("--- COMP70050 Decision Tree Coursework ---")
    
    print("\n--- Step 3: Evaluation (Before Pruning) ---\n")
    
    print("--- Clean Dataset ---")
    y_true_clean, y_pred_clean, avg_depth_clean = cross_validation(clean_dataset)
    cm_clean = confusion_matrix(y_true_clean, y_pred_clean)
    acc_clean, prec_clean, rec_clean, f1_clean = calculate_metrics(cm_clean)
    
    print("Confusion Matrix:\n", cm_clean)
    print(f"Accuracy: {acc_clean:.4f}")
    for i in range(len(prec_clean)):
        print(f"Class {i+1}: Precision={prec_clean[i]:.4f}, Recall={rec_clean[i]:.4f}, F1-Score={f1_clean[i]:.4f}")

    print("\n--- Noisy Dataset ---")
    y_true_noisy, y_pred_noisy, avg_depth_noisy = cross_validation(noisy_dataset)
    cm_noisy = confusion_matrix(y_true_noisy, y_pred_noisy)
    acc_noisy, prec_noisy, rec_noisy, f1_noisy = calculate_metrics(cm_noisy)
    
    print("Confusion Matrix:\n", cm_noisy)
    print(f"Accuracy: {acc_noisy:.4f}")
    for i in range(len(prec_noisy)):
        print(f"Class {i+1}: Precision={prec_noisy[i]:.4f}, Recall={rec_noisy[i]:.4f}, F1-Score={f1_noisy[i]:.4f}")
        
    print("\n\n--- Step 4: Evaluation (After Pruning) ---\n")

    print("--- Clean Dataset (Pruned) ---")
    y_true_clean_p, y_pred_clean_p, avg_depth_clean_p = nested_cross_validation(clean_dataset)
    cm_clean_p = confusion_matrix(y_true_clean_p, y_pred_clean_p)
    acc_clean_p, prec_clean_p, rec_clean_p, f1_clean_p = calculate_metrics(cm_clean_p)
    
    print("Confusion Matrix:\n", cm_clean_p)
    print(f"Accuracy: {acc_clean_p:.4f}")
    for i in range(len(prec_clean_p)):
        print(f"Class {i+1}: Precision={prec_clean_p[i]:.4f}, Recall={rec_clean_p[i]:.4f}, F1-Score={f1_clean_p[i]:.4f}")

    print("\n--- Noisy Dataset (Pruned) ---")
    y_true_noisy_p, y_pred_noisy_p, avg_depth_noisy_p = nested_cross_validation(noisy_dataset)
    cm_noisy_p = confusion_matrix(y_true_noisy_p, y_pred_noisy_p)
    acc_noisy_p, prec_noisy_p, rec_noisy_p, f1_noisy_p = calculate_metrics(cm_noisy_p)
    
    print("Confusion Matrix:\n", cm_noisy_p)
    print(f"Accuracy: {acc_noisy_p:.4f}")
    for i in range(len(prec_noisy_p)):
        print(f"Class {i+1}: Precision={prec_noisy_p[i]:.4f}, Recall={rec_noisy_p[i]:.4f}, F1-Score={f1_noisy_p[i]:.4f}")

    print("\n\n--- Depth Analysis ---")
    print(f"Clean Dataset - Avg. Depth Before Pruning: {avg_depth_clean:.2f}")
    print(f"Clean Dataset - Avg. Depth After Pruning: {avg_depth_clean_p:.2f}")
    print(f"Noisy Dataset - Avg. Depth Before Pruning: {avg_depth_noisy:.2f}")
    print(f"Noisy Dataset - Avg. Depth After Pruning: {avg_depth_noisy_p:.2f}")

    print("\n\n--- Bonus: Visualizing Tree for Entire Clean Dataset ---")
    full_clean_tree, _ = decision_tree_learning(clean_dataset)
    plot_tree(
        full_clean_tree,
        title="Decision Tree on Full Clean Dataset",
        title_pad=60,       # [修改] 拉大标题与树的距离
        node_fontsize=7    # [修改] 放大树中节点文字
    )

if __name__ == '__main__':
    main()
