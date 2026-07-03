import json
from pathlib import Path
from collections import defaultdict

def consolidate_results(base_dir: str = "results/shift_detection_simultaneous"):
    base_path = Path(base_dir)   
    consolidated_base_dir = base_path / "consolidated"
    consolidated_base_dir.mkdir(exist_ok=True)
    
    for node_size_dir in sorted(base_path.iterdir()):
        if not node_size_dir.is_dir() or node_size_dir.name == "consolidated":
            continue
            
        print(f"\nProcessing {node_size_dir.name}...")
        
        consolidated_node_dir = consolidated_base_dir / node_size_dir.name
        consolidated_node_dir.mkdir(exist_ok=True)
        
        for graph_type_dir in sorted(node_size_dir.iterdir()):
            if not graph_type_dir.is_dir():
                continue
                
            print(f"  Processing {graph_type_dir.name}...")
            
            consolidated_dir = consolidated_node_dir / graph_type_dir.name
            consolidated_dir.mkdir(exist_ok=True)
            dataset_files = defaultdict(list)
            
            for file_path in sorted(graph_type_dir.glob("result_dataset_*_node_*.json")):
                filename = file_path.name
                parts = filename.replace("result_dataset_", "").replace(".json", "").split("_node_")
                if len(parts) == 2:
                    dataset_num = int(parts[0])
                    dataset_files[dataset_num].append(file_path)
            
            for dataset_num, files in sorted(dataset_files.items()):
                consolidate_dataset(dataset_num, files, consolidated_dir)


def consolidate_dataset(dataset_num: int, files: list, output_dir: Path):
    node_results = {}
    dataset_info = None
    timestamps = []
    node_elapsed_times = {}
    total_elapsed_time = 0.0

    for file_path in files:
        with open(file_path, 'r') as f:
            data = json.load(f)

        node = data['node']
        elapsed_time = data.get('elapsed_time', None)
        node_elapsed_times[node] = elapsed_time
        if elapsed_time is not None:
            total_elapsed_time += float(elapsed_time)

        node_results[node] = {
            'true_shift': data['true_shift'],
            'shift_detected': data['shift_detected'],
            'correct': data['correct'],
            'H0': data['H0'],
            'H1': data['H1'],
            'H1.unequal': data['H1.unequal'],
            'elapsed_time': elapsed_time
        }

        if dataset_info is None:
            dataset_info = {
                'node_size': data['node_size'],
                'graph_type': data['graph_type'],
                'dataset': data['dataset']
            }

        if 'timestamp_start' in data:
            timestamps.append(data['timestamp_start'])
        if 'timestamp_end' in data:
            timestamps.append(data['timestamp_end'])

    sorted_nodes = sorted(node_results.keys())
    shifted_nodes = sorted([
        node for node, result in node_results.items()
        if result['true_shift'] in ['function', 'simultaneous']
    ])

    shift_types = {}
    for node in shifted_nodes:
        true_shift = node_results[node]['true_shift']
        if true_shift == 'function':
            shift_types[str(node)] = ['function']
        elif true_shift == 'simultaneous':
            shift_types[str(node)] = ['simultaneous']

    shift_case_parts = []
    for node in shifted_nodes:
        shift_case_parts.extend(shift_types[str(node)])
    shift_case = '_'.join(shift_case_parts) if shift_case_parts else 'no_shift'

    estimated_shift_types = {}
    for node in shifted_nodes:
        detected = node_results[node]['shift_detected']
        if detected in ['function', 'simultaneous']:
            estimated_shift_types[str(node)] = detected
        elif detected == 'none':
            estimated_shift_types[str(node)] = 'none'

    consolidated = {
        'dataset_info': {
            'dataset_index': dataset_num,
            'node_count': int(dataset_info['node_size'].replace('nodes_', '')),
            'graph_type': dataset_info['graph_type'],
            'shifted_nodes': shifted_nodes,
            'shift_types': shift_types,
            'shift_case': shift_case
        },
        'model': 'gpr',
        'node_results': {str(node): node_results[node] for node in sorted_nodes},
        'estimated_shift_types': estimated_shift_types,
        'node_elapsed_times': {str(node): node_elapsed_times[node] for node in sorted_nodes},
        'total_elapsed_time': total_elapsed_time
    }

    if timestamps:
        consolidated['timestamp_start'] = min(timestamps)
        consolidated['timestamp_end'] = max(timestamps)

    output_file = output_dir / f"result_dataset_{dataset_num}.json"
    with open(output_file, 'w') as f:
        json.dump(consolidated, f, indent=2)

    print(f"    Created: {output_file.name} (nodes: {sorted_nodes})")


def main():
    print("=" * 70)
    print("Consolidating FANS results from node-level to dataset-level")
    print("=" * 70)
    
    consolidate_results()
    
    print("\n" + "=" * 70)
    print("Consolidation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
