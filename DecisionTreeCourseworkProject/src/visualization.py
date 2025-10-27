import os
import matplotlib.pyplot as plt
from datetime import datetime


def plot_decision_tree(nodes_dict, save_name_prefix="decision_tree", show_fig=True):
    """
    Visualize the decision tree horizontally (left → right), formatted for A4 report pages.
    """

    # === 1️⃣ Build tree structure ===
    tree = {}
    for node_name, node_info in nodes_dict.items():
        if node_info["left"]:
            tree[node_name] = [node_info["left"], node_info["right"]]
        else:
            tree[node_name] = []

    # === 2️⃣ Helper functions ===
    def get_tree_depth(node):
        if nodes_dict[node]["leaf"][0]:
            return 1
        left, right = tree[node]
        return 1 + max(get_tree_depth(left), get_tree_depth(right))

    def get_num_leaves(node):
        if nodes_dict[node]["leaf"][0]:
            return 1
        left, right = tree[node]
        return get_num_leaves(left) + get_num_leaves(right)

    depth = get_tree_depth("n_0")
    total_leaves = get_num_leaves("n_0")

    # === 3️⃣ Assign positions ===
    node_positions = {}
    leaf_y = 0

    def assign_positions(node, depth_level):
        nonlocal leaf_y
        if nodes_dict[node]["leaf"][0]:
            node_positions[node] = (depth_level, leaf_y)
            leaf_y += 1
            return node_positions[node][1]

        left, right = tree[node]
        if left:
            ly = assign_positions(left, depth_level + 1)
        else:
            ly = leaf_y
        if right:
            ry = assign_positions(right, depth_level + 1)
        else:
            ry = leaf_y

        mid_y = (ly + ry) / 2
        node_positions[node] = (depth_level, mid_y)
        return mid_y

    assign_positions("n_0", 0)

    # Normalize y positions
    ys = [y for _, y in node_positions.values()]
    min_y, max_y = min(ys), max(ys)
    y_range = max_y - min_y if max_y > min_y else 1.0
    for k in node_positions:
        x, y = node_positions[k]
        node_positions[k] = (x, (y - min_y) / y_range)

    # === 4️⃣ Plot tree (A4-optimized) ===
    # Fit to A4 aspect ratio: width smaller, height larger
    fig_w = 12      # reduce width
    fig_h = 17 # increase height
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()

    fontsize = 12
    pad_scale = 0.6
    line_width = 1.2
    x_spacing = 0.9 / depth  # tighter horizontal distance

    def plot_node(node):
        x, y = node_positions[node]
        node_info = nodes_dict[node]

        if node_info["leaf"][0]:
            label = f"Leaf: {node_info['leaf'][1]}"
            box_color = "#B4E7B0"
        else:
            label = f"X[{node_info['attribute']}] < {node_info['value']:.2f}"
            box_color = "#A6C8E0"

        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            bbox=dict(boxstyle=f"round,pad={pad_scale}", fc=box_color, ec="black", lw=line_width),
            fontsize=fontsize,
        )

        if not node_info["leaf"][0]:
            for child in [node_info["left"], node_info["right"]]:
                if child in node_positions:
                    cx, cy = node_positions[child]
                    ax.plot(
                        [x + 0.2, cx - 0.2],  # horizontal connectors
                        [y, cy],
                        "k-",
                        lw=line_width * 0.8,
                    )
                    plot_node(child)

    plot_node("n_0")

    plt.title("Decision Tree Visualization", fontsize=fontsize + 3, pad=10)
    plt.tight_layout()

    # === 5️⃣ Save ===
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{save_name_prefix}_{timestamp}.png"

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    save_dir = os.path.join(base_dir, "result", "visual_tree")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ Decision tree image saved to: {save_path}")

    if show_fig:
        plt.show()
    else:
        plt.close(fig)
