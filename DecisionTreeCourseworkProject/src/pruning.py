"""Post-pruning utilities for the decision tree.

This module implements Reduced Error Pruning operating on the existing
`nodes_dict` tree representation used by the project. The functions here
work directly with the dictionary representation and with the project's
DecisionTree class for predictions.
"""

from typing import List
import copy
import numpy as np


def get_internal_nodes(nodes_dict: dict) -> List[str]:
	"""Return all internal node keys in the tree.

	An internal node is any node whose 'leaf' flag is False.

	Args:
		nodes_dict: dictionary representation of the tree.

	Returns:
		A list of node keys (e.g. ['n_0', 'n_01', ...]) that are internal.
	"""
	internal = [k for k, v in nodes_dict.items() if not v['leaf'][0]]
	return internal


def is_prunable(node_key: str, nodes_dict: dict) -> bool:
	"""Determine whether a node is prunable.

	A node is considered prunable when both its left and right children
	are leaf nodes. If the node itself is a leaf or children are missing
	the function returns False.

	Args:
		node_key: key of the node to check.
		nodes_dict: tree dictionary.

	Returns:
		True if both children exist and are leaves; False otherwise.
	"""
	if node_key not in nodes_dict:
		return False
	node_info = nodes_dict[node_key]
	# if current node is already a leaf, it's not an internal node
	if node_info['leaf'][0]:
		return False

	left = node_info.get('left')
	right = node_info.get('right')
	if left is None or right is None:
		return False
	if left not in nodes_dict or right not in nodes_dict:
		return False

	return nodes_dict[left]['leaf'][0] and nodes_dict[right]['leaf'][0]


def _indices_reaching_node(nodes_dict: dict, node_key: str, X: np.ndarray) -> np.ndarray:
	"""Return boolean indices of rows in X that would reach node_key.

	This simulates routing each sample from the root and records which
	samples end up at the requested node.
	"""
	n_samples = X.shape[0]
	indices = np.zeros(n_samples, dtype=bool)
	for i in range(n_samples):
		cur = 'n_0'
		x = X[i]
		# walk until leaf or the requested node
		while True:
			if cur == node_key:
				indices[i] = True
				break
			info = nodes_dict[cur]
			if info['leaf'][0]:
				# arrived at leaf that is not the node_key
				break
			attr = info['attribute']
			val = info['value']
			if x[attr] < val:
				cur = info['left']
			else:
				cur = info['right']
			# safety: if navigation goes to a missing key, stop
			if cur not in nodes_dict:
				break
	return indices


def prune_tree(nodes_dict: dict, DecisionTree, X_train: np.ndarray, y_train: np.ndarray,
			   X_val: np.ndarray, y_val: np.ndarray) -> dict:
	"""Perform reduced-error post-pruning on the provided tree.

	The algorithm iteratively considers internal nodes that have leaf
	children (i.e. prunable). For each prunable node it replaces the
	subtree by a leaf with the majority label of the training samples
	that reach that node. The replacement is kept only if it improves
	accuracy on the validation set.

	Args:
		nodes_dict: original tree dictionary to prune (will not be mutated).
		DecisionTree: the DecisionTree class (used to create a predictor).
		X_train, y_train: training data used to determine majority labels
			for nodes being pruned.
		X_val, y_val: validation data used to evaluate pruning decisions.

	Returns:
		A new nodes_dict dictionary representing the pruned tree.
	"""
	pruned_nodes = copy.deepcopy(nodes_dict)

	# validator that will be used to evaluate accuracy on validation set
	# when predicting we need a DecisionTree instance whose train_y exists
	# so predictions have the correct dtype
	while True:
		validator = DecisionTree(X_train, y_train, X_val)
		# baseline accuracy with current pruned_nodes
		y_val_pred = validator.predict(pruned_nodes)
		best_accuracy = float(np.mean(y_val_pred == y_val))
		best_candidate_tree = None

		internal_nodes = get_internal_nodes(pruned_nodes)

		for node_key in internal_nodes:
			if not is_prunable(node_key, pruned_nodes):
				continue

			# simulate pruning at node_key
			temp_tree = copy.deepcopy(pruned_nodes)

			# find training samples that reach node_key to compute majority label
			idx_mask = _indices_reaching_node(temp_tree, node_key, X_train)
			if np.any(idx_mask):
				labels_here = y_train[idx_mask]
				unique, counts = np.unique(labels_here, return_counts=True)
				majority_label = unique[np.argmax(counts)]
			else:
				# fallback: use majority of the two child leaves' labels
				left = temp_tree[node_key]['left']
				right = temp_tree[node_key]['right']
				left_label = temp_tree[left]['leaf'][1]
				right_label = temp_tree[right]['leaf'][1]
				# simple majority between two labels (if tie, pick left)
				majority_label = left_label if left_label == right_label else left_label

			# convert node_key to a leaf in temp_tree
			temp_tree[node_key]['leaf'] = (True, int(majority_label))
			temp_tree[node_key]['attribute'] = None
			temp_tree[node_key]['value'] = None
			temp_tree[node_key]['left'] = None
			temp_tree[node_key]['right'] = None

			# evaluate on validation set
			validator_temp = DecisionTree(X_train, y_train, X_val)
			y_val_pred_temp = validator_temp.predict(temp_tree)
			acc_temp = float(np.mean(y_val_pred_temp == y_val))

			if acc_temp > best_accuracy:
				best_accuracy = acc_temp
				best_candidate_tree = temp_tree

		if best_candidate_tree is not None:
			pruned_nodes = best_candidate_tree
			# continue the outer loop to search for further improvements
			continue
		else:
			# no improvement found, finish
			break

	return pruned_nodes
