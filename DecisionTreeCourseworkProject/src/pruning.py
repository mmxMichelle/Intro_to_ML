"""Post-pruning utilities for the decision tree.

This module implements Reduced Error Pruning operating on the existing
`nodes_dict` tree representation used by the project. The functions here
work directly with the dictionary representation and with the project's
DecisionTree class for predictions.
"""

from typing import List, Dict
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


def compute_node_reach_indices(nodes_dict: dict, X: np.ndarray) -> Dict[str, np.ndarray]:
	"""
	For every node key in nodes_dict, compute the indices of rows in X whose
	path from the root includes that node. Returns a mapping node_key -> np.array(indices).
	Complexity: O(n_samples * depth) — done ONCE per pass.
	"""
	n = X.shape[0]
	node_to_indices = {k: [] for k in nodes_dict.keys()}  # build lists then convert
	for i in range(n):
		cur = 'n_0'
		x = X[i]
		# walk until leaf or missing
		while True:
			# record that sample i visits cur node
			if cur in node_to_indices:
				node_to_indices[cur].append(i)
			else:
				# safety: if tree has broken keys, stop
				break
			info = nodes_dict[cur]
			if info['leaf'][0]:
				break
			attr = info['attribute']
			val = info['value']
			# if attr is None (shouldn't be for internal nodes), break
			if attr is None:
				break
			if x[attr] < val:
				cur = info['left']
			else:
				cur = info['right']
			if cur not in nodes_dict:
				break
	# convert lists to numpy arrays (empty lists -> empty arrays)
	for k, lst in node_to_indices.items():
		node_to_indices[k] = np.array(lst, dtype=int) if lst else np.zeros(0, dtype=int)
	return node_to_indices


def prune_tree(nodes_dict: dict, DecisionTree, X_train: np.ndarray, y_train: np.ndarray,
			   X_val: np.ndarray, y_val: np.ndarray) -> dict:
	"""
	Reduced-error pruning optimized:
	- precompute which samples reach each node (train+val)
	- compute baseline predictions on val once per pass
	- evaluate a candidate by adjusting only those val preds that reach the candidate node
	"""
	pruned_nodes = copy.deepcopy(nodes_dict)

	while True:
		# Precompute sample→node visits for pruned_nodes
		train_node_indices = compute_node_reach_indices(pruned_nodes, X_train)
		val_node_indices = compute_node_reach_indices(pruned_nodes, X_val)

		# validator only used for dtype etc. We don't call predict repeatedly now.
		validator = DecisionTree(X_train, y_train, X_val)
		# baseline predictions once for this pass
		y_val_pred = validator.predict(pruned_nodes)
		baseline_accuracy = float(np.mean(y_val_pred == y_val))

		current_pass_best_acc = baseline_accuracy
		best_candidate = None  # will hold (node_key, majority_label) if found

		# iterate internal nodes once
		internal_nodes = get_internal_nodes(pruned_nodes)

		for node_key in internal_nodes:
			if not is_prunable(node_key, pruned_nodes):
				continue

			# find training samples that reach node_key to compute majority label
			train_idx = train_node_indices.get(node_key, np.zeros(0, dtype=int))
			if train_idx.size > 0:
				labels_here = y_train[train_idx]
				# use bincount if labels are non-negative integers and small range
				if np.issubdtype(labels_here.dtype, np.integer) and labels_here.size > 0:
					min_label = labels_here.min()
					if min_label >= 0:
						# shift to zero-based if necessary
						shift = 0
						if min_label != 0:
							labels_shifted = labels_here - min_label
							counts = np.bincount(labels_shifted)
							majority_label = np.argmax(counts) + min_label
						else:
							counts = np.bincount(labels_here)
							majority_label = np.argmax(counts)
					else:
						# fallback
						unique, counts = np.unique(labels_here, return_counts=True)
						majority_label = unique[np.argmax(counts)]
				else:
					unique, counts = np.unique(labels_here, return_counts=True)
					majority_label = unique[np.argmax(counts)]
			else:
				# fallback: majority of the two child leaves' labels
				left = pruned_nodes[node_key]['left']
				right = pruned_nodes[node_key]['right']
				left_label = pruned_nodes[left]['leaf'][1]
				right_label = pruned_nodes[right]['leaf'][1]
				majority_label = left_label if left_label == right_label else left_label

			# determine which val samples are affected by pruning this node
			val_idx = val_node_indices.get(node_key, np.zeros(0, dtype=int))
			if val_idx.size == 0:
				# if no val samples reach node, pruning won't change val accuracy;
				# but you may still want to treat it as improvement only if baseline unchanged.
				acc_temp = baseline_accuracy
			else:
				# simulate effect: only change predictions for these indices
				y_temp = y_val_pred.copy()
				y_temp[val_idx] = majority_label
				acc_temp = float(np.mean(y_temp == y_val))

			# keep best (>= as original)
			if acc_temp >= current_pass_best_acc:
				current_pass_best_acc = acc_temp
				best_candidate = (node_key, int(majority_label))

		if best_candidate is not None:
			# apply the single best candidate to pruned_nodes (mutate in-place)
			node_key, maj_label = best_candidate
			pruned_nodes[node_key]['leaf'] = (True, int(maj_label))
			pruned_nodes[node_key]['attribute'] = None
			pruned_nodes[node_key]['value'] = None
			pruned_nodes[node_key]['left'] = None
			pruned_nodes[node_key]['right'] = None
			# loop again (recompute node-to-sample maps)
			continue
		else:
			break

	return pruned_nodes

def compute_average_depth(nodes_dict: dict, root_key: str = 'n_0') -> float:
	"""Compute the average depth of all leaf nodes in the tree."""
	def _dfs(node_key: str, depth: int, depths: list):
		node = nodes_dict[node_key]
		if node['leaf'][0]:
			depths.append(depth)
			return
		if node.get('left'):
			_dfs(node['left'], depth + 1, depths)
		if node.get('right'):
			_dfs(node['right'], depth + 1, depths)
	depths = []
	_dfs(root_key, 0, depths)
	return float(np.mean(depths)) if depths else 0.0

