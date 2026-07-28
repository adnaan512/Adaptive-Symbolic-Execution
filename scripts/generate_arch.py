import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_box(ax, center_x, center_y, text, width=4, height=1):
    box = patches.FancyBboxPatch(
        (center_x - width/2, center_y - height/2), 
        width, height,
        boxstyle="round,pad=0.2",
        edgecolor='#2c3e50',
        facecolor='#ecf0f1',
        lw=2
    )
    ax.add_patch(box)
    ax.text(center_x, center_y, text, ha='center', va='center', fontsize=12, fontweight='bold', color='#2c3e50', family='sans-serif')

def draw_arrow(ax, start_x, start_y, end_x, end_y):
    ax.annotate(
        '', xy=(end_x, end_y), xytext=(start_x, start_y),
        arrowprops=dict(arrowstyle="->", color='#34495e', lw=2)
    )

def main():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.axis('off')

    # Coordinates
    nodes = {
        "C Program": (5, 12),
        "LLVM Bitcode": (5, 10.5),
        "KLEE Symbolic Execution": (5, 9),
        "Execution States": (5, 7.5),
        "Feature Extraction": (5, 6),
        "Machine Learning": (3, 4.5),
        "LLM Analysis": (7, 4.5),
        "Reinforcement Learning": (5, 3),
        "Priority Ranking": (5, 1.5),
        "Improved Branch Coverage": (5, 0)
    }

    # Draw nodes
    for name, (x, y) in nodes.items():
        draw_box(ax, x, y, name, width=4, height=0.8)

    # Draw edges
    edges = [
        ("C Program", "LLVM Bitcode"),
        ("LLVM Bitcode", "KLEE Symbolic Execution"),
        ("KLEE Symbolic Execution", "Execution States"),
        ("Execution States", "Feature Extraction"),
        ("Feature Extraction", "Machine Learning"),
        ("Feature Extraction", "LLM Analysis"),
        ("Machine Learning", "Reinforcement Learning"),
        ("LLM Analysis", "Reinforcement Learning"),
        ("Reinforcement Learning", "Priority Ranking"),
        ("Priority Ranking", "Improved Branch Coverage")
    ]

    for start, end in edges:
        start_pos = nodes[start]
        end_pos = nodes[end]
        # Adjust arrow start/end to touch box edges
        draw_arrow(ax, start_pos[0], start_pos[1] - 0.4, end_pos[0], end_pos[1] + 0.4)

    plt.title("Adaptive Symbolic Execution Architecture", fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('paper/figures/architecture.png', dpi=300, bbox_inches='tight')
    print("Saved architecture diagram to paper/figures/architecture.png")

if __name__ == "__main__":
    main()
