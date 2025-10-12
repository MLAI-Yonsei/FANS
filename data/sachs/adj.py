import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
from pathlib import Path

# Find project root based on current file location
current_file = Path(__file__)  
project_root = current_file.parent.parent.parent 

# Read CSV file
file_path = project_root / 'sachs/GroundTruth2.csv'
df = pd.read_csv(file_path)

# Extract all nodes
all_nodes = set(df['from'].tolist() + df['to'].tolist())
print(f"Total nodes: {len(all_nodes)}")
print(f"Nodes: {sorted(all_nodes)}")

# Create graph structure for topological sort
edges = [(row['from'], row['to']) for _, row in df.iterrows()]
print(f"Edges: {edges}")

# Check for cycles using NetworkX
G_check = nx.DiGraph()
G_check.add_edges_from(edges)
print(f"Is DAG: {nx.is_directed_acyclic_graph(G_check)}")

# Perform topological sort (Kahn's algorithm)
from collections import defaultdict, deque

# Calculate graph and in-degree
graph = defaultdict(list)
in_degree = defaultdict(int)

# Initialize all nodes
for node in all_nodes:
    in_degree[node] = 0

# Add edges and calculate in-degree
for from_node, to_node in edges:
    graph[from_node].append(to_node)
    in_degree[to_node] += 1

# Topological sort
queue = deque([node for node in all_nodes if in_degree[node] == 0])
topological_order = []

while queue:
    node = queue.popleft()
    topological_order.append(node)
    
    for neighbor in graph[node]:
        in_degree[neighbor] -= 1
        if in_degree[neighbor] == 0:
            queue.append(neighbor)

print(f"Topological order: {topological_order}")

# Check if all nodes are included
if len(topological_order) != len(all_nodes):
    missing_nodes = all_nodes - set(topological_order)
    print(f"Warning: Missing nodes in topological order: {missing_nodes}")
    print("This indicates cycles in the graph")
    # Add missing nodes at the end
    topological_order.extend(list(missing_nodes))

# Map nodes to indices
node_to_index = {node: i for i, node in enumerate(topological_order)}
print(f"Node to index mapping: {node_to_index}")

# Create 11x11 adjacency matrix
adj_matrix = np.zeros((11, 11), dtype=int)

# Fill matrix according to edges
for from_node, to_node in edges:
    i = node_to_index[from_node]
    j = node_to_index[to_node]
    adj_matrix[i, j] = 1

print("Adjacency Matrix:")
print(adj_matrix)

# Check if upper triangular
is_upper_triangular = np.allclose(adj_matrix, np.triu(adj_matrix))
print(f"Is upper triangular: {is_upper_triangular}")

# Save to adj.npy file
np.save(project_root / 'sachs/adj.npy', adj_matrix)
print("adj.npy file has been created.")

# Validation: Load and verify saved file
loaded_adj = np.load(project_root / 'sachs/adj.npy')
print("Loaded adjacency matrix:")
print(loaded_adj)

# DAG graph visualization (improved version)
def visualize_dag():
    # Create NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes (display index and name together)
    for i, node in enumerate(topological_order):
        G.add_node(node, index=i)
    
    # Add edges
    for from_node, to_node in edges:
        G.add_edge(from_node, to_node)
    
    # Set graph layout (hierarchical layout considering topological order)
    plt.figure(figsize=(16, 12))
    
    # Set hierarchical positions based on topological order
    pos = {}
    levels = defaultdict(list)
    
    # Calculate level of each node (based on longest path)
    node_levels = {}
    for node in topological_order:
        max_level = 0
        for pred in G.predecessors(node):
            if pred in node_levels:
                max_level = max(max_level, node_levels[pred] + 1)
        node_levels[node] = max_level
        levels[max_level].append(node)
    
    # Calculate positions (expand spacing between nodes)
    for level, nodes in levels.items():
        for i, node in enumerate(nodes):
            x = (i - (len(nodes) - 1) / 2) * 2.5  # Expand spacing
            y = -level * 2.0  # Expand level spacing
            pos[node] = (x, y)
    
    # Create node labels (index and name together)
    labels = {node: f"{node_to_index[node]}\n{node}" for node in topological_order}
    
    # Draw nodes (improved size and color)
    nx.draw_networkx_nodes(G, pos, 
                          node_color='lightcoral', 
                          node_size=3500,
                          alpha=0.9,
                          edgecolors='black',
                          linewidths=2)
    
    # Draw edges (improved arrows)
    nx.draw_networkx_edges(G, pos, 
                          edge_color='darkblue',
                          arrows=True,
                          arrowsize=30,  # Increase arrow size
                          arrowstyle='-|>',  # Change arrow style
                          width=3,  # Increase edge thickness
                          alpha=0.8,
                          connectionstyle="arc3,rad=0.1")  # Slightly curved
    
    # Draw labels (increase font size)
    nx.draw_networkx_labels(G, pos, labels, 
                           font_size=11,
                           font_weight='bold',
                           font_color='white')
    
    plt.title("DAG Visualization with Enhanced Arrows\n(Index: NodeName)", 
              fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    
    # Add legend
    plt.text(0.02, 0.98, 'Arrow Direction: Causal Flow', 
             transform=plt.gca().transAxes, 
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
             verticalalignment='top', fontsize=12)
    
    plt.tight_layout()
    
    # Save graph (high resolution)
    plt.savefig(project_root / 'sachs/dag_visualization.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print additional information
    print("\n=== DAG Information ===")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Is DAG: {nx.is_directed_acyclic_graph(G)}")
    print(f"Topological sort: {list(nx.topological_sort(G))}")
    
    # Print node index and name mapping table
    print("\n=== Node Index Mapping ===")
    for node in topological_order:
        print(f"Index {node_to_index[node]:2d}: {node}")

# Alternative visualization (sharper arrows)
def visualize_dag_alternative():
    """Alternative visualization method - sharper arrows"""
    G = nx.DiGraph()
    
    for i, node in enumerate(topological_order):
        G.add_node(node, index=i)
    
    for from_node, to_node in edges:
        G.add_edge(from_node, to_node)
    
    plt.figure(figsize=(18, 14))
    
    # Use spring layout (more natural positioning)
    pos = nx.spring_layout(G, k=3, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, 
                          node_color='skyblue', 
                          node_size=4000,
                          alpha=0.9,
                          edgecolors='navy',
                          linewidths=3)
    
    # Draw each edge individually (sharper arrows)
    for edge in G.edges():
        start_pos = pos[edge[0]]
        end_pos = pos[edge[1]]
        
        plt.annotate('', xy=end_pos, xytext=start_pos,
                    arrowprops=dict(arrowstyle='->', 
                                  color='red',
                                  lw=4,  # Line thickness
                                  shrinkA=25, shrinkB=25,  # Slightly away from nodes
                                  mutation_scale=25))  # Arrow size
    
    # Draw labels
    labels = {node: f"{node_to_index[node]}\n{node}" for node in topological_order}
    nx.draw_networkx_labels(G, pos, labels, 
                           font_size=12,
                           font_weight='bold')
    
    plt.title("DAG Visualization - Alternative Style\n(Red Arrows Show Causal Direction)", 
              fontsize=18, fontweight='bold', pad=20)
    plt.axis('off')
    plt.tight_layout()
    
    plt.savefig(project_root / 'sachs/dag_visualization.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

# Run both visualizations
print("\n" + "="*50)
print("Visualizing DAG graph...")
visualize_dag()

print("\n" + "="*50)
print("Visualizing DAG graph in alternative style...")
visualize_dag_alternative()