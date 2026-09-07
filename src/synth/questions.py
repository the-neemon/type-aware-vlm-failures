def build_bar_questions(categories, values):
    """Build grounded structural prompts and one absent-category prompt."""
    first_category, second_category = categories[:2]
    first_value, second_value = values[:2]
    larger_category = first_category if first_value > second_value else second_category
    missing_category = next(
        (
            candidate
            for candidate in (chr(code) for code in range(ord("A"), ord("Z") + 1))
            if candidate not in categories
        ),
        "MISSING",
    )
    return [
        {
            "template": "read_value",
            "target_failure_type": "structural",
            "question": f"What is the value of category {first_category}?",
            "gold_answer": first_value,
        },
        {
            "template": "compare_bars",
            "target_failure_type": "structural",
            "question": f"Which category is larger, {first_category} or {second_category}?",
            "gold_answer": larger_category,
        },
        {
            "template": "bar_difference",
            "target_failure_type": "structural",
            "question": f"What is the absolute difference between {first_category} and {second_category}?",
            "gold_answer": abs(first_value - second_value),
        },
        {
            "template": "absent_category",
            "target_failure_type": "fabrication",
            "question": f"What is the value of category {missing_category}?",
            "gold_answer": "not present",
        },
    ]


def build_node_questions(path_nodes):
    """Build path-reading prompts and one prompt about an absent node."""
    start_node, end_node = path_nodes[0], path_nodes[-1]
    return [
        {
            "template": "shortest_path_length",
            "target_failure_type": "structural",
            "question": f"How many edges are in the shortest path from {start_node} to {end_node}?",
            "gold_answer": len(path_nodes) - 1,
        },
        {
            "template": "path_successor",
            "target_failure_type": "structural",
            "question": f"Which node comes immediately after {start_node} on the path to {end_node}?",
            "gold_answer": path_nodes[1],
        },
        {
            "template": "absent_node",
            "target_failure_type": "fabrication",
            "question": f"Is node MISSING connected to {start_node}?",
            "gold_answer": "node MISSING is not present",
        },
    ]
