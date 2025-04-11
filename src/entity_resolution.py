import sklearn as sk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import time
import datetime
import random
import math
import re
import json
import copy
import sqlite3
import ast
from tqdm import tqdm
import networkx as nx

# Set pandas display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

def set_project_dir():
    """Set the working directory to the project root."""
    project_name = 'ChatPD'
    os.chdir(os.getcwd()[:os.getcwd().find(project_name) + len(project_name)])
    print(f"Working directory set to: {os.getcwd()}")

def preprocess_data():
    """Load and preprocess the raw entity resolution data."""
    # Read the raw data
    er_raw_data_path = 'data/er_evaluation/er_raw_data.json'
    data_descriptions = pd.read_json(er_raw_data_path, orient='records', lines=True)
    print(f"Loaded data columns: {data_descriptions.columns}")
    
    # Convert to dict format
    data_descriptions = data_descriptions.to_dict(orient='records')
    return data_descriptions

def load_pwc_entities():
    """Load and sort Papers With Code dataset entities."""
    pwc_dataset_entities_path = 'data/pwc_data/250204/datasets.json'
    pwc_dataset_entities = json.load(open(pwc_dataset_entities_path, 'r'))
    pwc_dataset_entities = sorted(pwc_dataset_entities, key=lambda x: x['num_papers'], reverse=True)
    return pwc_dataset_entities

def build_pwc_graph(pwc_dataset_entities):
    """Build a graph of Papers With Code dataset entities."""
    PWC_G = nx.DiGraph()
    entity_num_papers_dict = {f"{dataset['name']}_$entity$": dataset['num_papers'] for dataset in pwc_dataset_entities}

    # Create a set of unique name nodes
    unique_name_nodes = set()
    for dataset in pwc_dataset_entities:
        unique_name_nodes.add(dataset['name'])
        if dataset['full_name']:
            unique_name_nodes.add(dataset['full_name'])

    # Add nodes
    for dataset in pwc_dataset_entities:
        dataset['entity_name'] = f"{dataset['name']}_$entity$"
        PWC_G.add_node(dataset['entity_name'], type='entity')
        PWC_G.add_node(dataset['name'], type='name')
        PWC_G.add_node(dataset['name'].lower(), type='name')
        PWC_G.add_node(dataset['name'].replace(' ', '-'), type='name')
        if dataset['full_name']:
            PWC_G.add_node(dataset['full_name'], type='name')
            PWC_G.add_node(dataset['full_name'].lower(), type='name')
            PWC_G.add_node(dataset['full_name'].replace(' ', '-'), type='name')
        for variant in dataset['variants']:
            PWC_G.add_node(variant, type='name')
            PWC_G.add_node(variant.lower(), type='name')
            PWC_G.add_node(variant.replace(' ', '-'), type='name')
        if dataset['homepage']:
            PWC_G.add_node(dataset['homepage'], type='url')

    # Add "has" edges
    for dataset in pwc_dataset_entities:
        PWC_G.add_edge(dataset['entity_name'], dataset['name'], relation='has_name')
        PWC_G.add_edge(dataset['entity_name'], dataset['name'].lower(), relation='has_name')
        PWC_G.add_edge(dataset['entity_name'], dataset['name'].replace(' ', '-'), relation='has_name')
        if dataset['full_name']:
            PWC_G.add_edge(dataset['entity_name'], dataset['full_name'], relation='has_name')
            PWC_G.add_edge(dataset['entity_name'], dataset['full_name'].lower(), relation='has_name')
            PWC_G.add_edge(dataset['entity_name'], dataset['full_name'].replace(' ', '-'), relation='has_name')
        for variant in dataset['variants']:
            PWC_G.add_edge(dataset['entity_name'], variant, relation='has_name')
            PWC_G.add_edge(dataset['entity_name'], variant.lower(), relation='has_name')
            PWC_G.add_edge(dataset['entity_name'], variant.replace(' ', '-'), relation='has_name')
        if dataset['homepage']:
            PWC_G.add_edge(dataset['entity_name'], dataset['homepage'], relation='has_url')

    return PWC_G

def add_edge_with_check(G, source, target, relation, dataset):
    """Add an edge to the graph with a check."""
    G.add_edge(source, target, relation=relation)

def add_edge_with_remove(G, source, target, relation, dataset):
    """Remove all outgoing edges from source before adding a new edge."""
    # Remove all outgoing edges from source
    for edge in list(G.out_edges(source)):
        G.remove_edge(edge[0], edge[1])
    G.add_edge(source, target, relation=relation)

def add_refers_to_edges(PWC_G, pwc_dataset_entities, unique_name_nodes):
    """Add 'refers_to' edges to the graph."""
    print('Start to add name refers_to edges')
    for dataset in pwc_dataset_entities:
        add_edge_with_check(PWC_G, dataset['name'], dataset['entity_name'], 'refers_to', dataset)
        add_edge_with_check(PWC_G, dataset['name'].lower(), dataset['entity_name'], 'refers_to', dataset)
        add_edge_with_check(PWC_G, dataset['name'].replace(' ', '-'), dataset['entity_name'], 'refers_to', dataset)
    
    print('Start to add full_name refers_to edges')
    for dataset in pwc_dataset_entities:
        if dataset['full_name'] and dataset['full_name'] != dataset['name']:
            add_edge_with_check(PWC_G, dataset['full_name'], dataset['entity_name'], 'refers_to', dataset)
            add_edge_with_check(PWC_G, dataset['full_name'].lower(), dataset['entity_name'], 'refers_to', dataset)
            add_edge_with_check(PWC_G, dataset['full_name'].replace(' ', '-'), dataset['entity_name'], 'refers_to', dataset)
    
    print('Start to add variants refers_to edges')
    for dataset in pwc_dataset_entities:
        for variant in dataset['variants']:
            if variant not in unique_name_nodes:
                add_edge_with_check(PWC_G, variant, dataset['entity_name'], 'refers_to', dataset)
                add_edge_with_check(PWC_G, variant.lower(), dataset['entity_name'], 'refers_to', dataset)
                add_edge_with_check(PWC_G, variant.replace(' ', '-'), dataset['entity_name'], 'refers_to', dataset)
    
    print('Start to add url refers_to edges')
    for dataset in pwc_dataset_entities:
        if dataset['homepage']:
            add_edge_with_check(PWC_G, dataset['homepage'], dataset['entity_name'], 'refers_to', dataset)

def filter_data_descriptions(data_descriptions):
    """Filter out meaningless dataset descriptions."""
    # Define lists of meaningless entries
    common_meanless_nodes = [
        "N/A", "NA", "Unknown", "Not explicitly mentioned", "unknown",
        "Not provided.", "Not applicable", "Open-sourced, URL not provided",
        "Not publicly available", "Not explicitly named", "(not explicitly provided)",
        "Not specified",
    ]
    
    meanless_names = [
        "Dataset-I", "Dataset I", "Dataset-II", "Dataset II",
        "Dataset 1", "Dataset1", "Dataset 2", "Dataset2", "Dataset 3", "Dataset3",
        "Dataset 4", "Dataset4", "Dataset 5", "Dataset5", "Dataset 6", "Dataset6",
        "Dataset A", "Dataset B", "Test Dataset", "Training Dataset",
        "Synthetic", "Synthetic Dataset", "Synthetic Datasets", "Synthetic Data",
        "Synthetic-1", "Synthetic-2", "Real Dataset", "Evaluation Dataset",
        "Real-World Dataset", "Real-world Dataset", "Real-World Datasets",
        "Real-World Datasets", "Real Data Datasets", "Simulated Data",
        "Simulated Dataset", "Simulated Datasets", "Various Classification Datasets",
        "Various Public Datasets",
    ]
    
    meanless_urls = ["URL not provided"]
    
    meanless_names += common_meanless_nodes
    meanless_urls += common_meanless_nodes
    common_meanless_nodes = [node.lower() for node in common_meanless_nodes]
    meanless_names = [name.lower() for name in meanless_names]
    meanless_urls = [url.lower() for url in meanless_urls]

    # Filter out meaningless dataset descriptions
    filtered_data_descriptions = []
    for dataset_description in data_descriptions:
        if (
            dataset_description.get("dataset name")
            and dataset_description["dataset name"].lower() not in meanless_names
            and "not specified" not in dataset_description["dataset name"].lower()
            and "not provided" not in dataset_description["dataset name"].lower()
        ):
            filtered_data_descriptions.append(dataset_description)

    # Clean URLs
    for dataset_description in filtered_data_descriptions:
        if isinstance(dataset_description.get("dataset url"), list):
            dataset_description["dataset url"] = str(dataset_description["dataset url"])
        
        # Remove meaningless URLs
        if (
            dataset_description.get("dataset url")
            and (
                dataset_description["dataset url"].lower() in meanless_urls
                or "not specified" in dataset_description["dataset url"].lower()
                or "not provided" in dataset_description["dataset url"].lower()
                or "not available" in dataset_description["dataset url"].lower()
                or "unspecified" in dataset_description["dataset url"].lower()
                or "unknown" in dataset_description["dataset url"].lower()
                or "not publicly available" in dataset_description["dataset url"].lower()
                or not (
                    dataset_description["dataset url"].startswith("http://")
                    or dataset_description["dataset url"].startswith("https://")
                    or dataset_description["dataset url"].startswith("www")
                )
            )
        ):
            dataset_description.pop("dataset url")

    # Count names
    name_value_counts = pd.Series(
        [
            dataset_description["dataset name"]
            for dataset_description in filtered_data_descriptions
            if dataset_description.get("dataset name")
        ]
    ).value_counts()
    
    print(f"Total number of unique dataset names: {len(name_value_counts)}")
    
    return filtered_data_descriptions

def build_mixed_graph(PWC_G, filtered_data_descriptions):
    """Build a mixed graph combining PWC entities and dataset descriptions."""
    Mixed_G = nx.DiGraph()
    
    # Copy nodes and edges from PWC_G
    for node in PWC_G.nodes():
        Mixed_G.add_node(node, type=PWC_G.nodes[node]['type'])
    
    for edge in PWC_G.edges(data=True):
        Mixed_G.add_edge(edge[0], edge[1], relation=edge[2]['relation'])

    # Add dataset description nodes
    for dataset_description in filtered_data_descriptions:
        dataset_description['entity_name'] = f"{dataset_description}"
        Mixed_G.add_node(dataset_description['entity_name'], type='dataset_description')

    # Add name nodes
    for dataset_description in filtered_data_descriptions:
        if dataset_description.get('dataset name'):
            Mixed_G.add_node(dataset_description['dataset name'], type='name')
        
        # Extract names in brackets
        if dataset_description.get('dataset name') and '(' in dataset_description['dataset name'] and ')' in dataset_description['dataset name']:
            name_in_brackets = dataset_description['dataset name'][
                dataset_description['dataset name'].find('(') + 1:
                dataset_description['dataset name'].find(')')
            ]
            Mixed_G.add_node(name_in_brackets, type='name')
        
        if dataset_description.get('dataset url'):
            Mixed_G.add_node(dataset_description['dataset url'], type='url')
    
    # Add has edges
    for dataset_description in filtered_data_descriptions:
        if dataset_description.get('dataset name'):
            Mixed_G.add_edge(dataset_description['entity_name'], dataset_description['dataset name'], relation='has_name')
        
        # Connect to names in brackets
        if dataset_description.get('dataset name') and '(' in dataset_description['dataset name'] and ')' in dataset_description['dataset name']:
            name_in_brackets = dataset_description['dataset name'][
                dataset_description['dataset name'].find('(') + 1:
                dataset_description['dataset name'].find(')')
            ]
            Mixed_G.add_edge(dataset_description['entity_name'], name_in_brackets, relation='has_name')
        
        if dataset_description.get('dataset url'):
            Mixed_G.add_edge(dataset_description['entity_name'], dataset_description['dataset url'], relation='has_url')
    
    return Mixed_G

def apply_rules(Mixed_G):
    """Apply rules to resolve entities in the graph."""
    # Create a mapping of lowercase node names
    lower_case_nodes = {}
    for n in Mixed_G.nodes():
        if isinstance(n, str):
            if n.lower() in lower_case_nodes:
                lower_case_nodes[n.lower()].append(n)
            else:
                lower_case_nodes[n.lower()] = [n]
    
    for node in Mixed_G.nodes():
        if not isinstance(node, str):
            continue
            
        
        # General rules
        # Rule for nodes with content in brackets
        if not Mixed_G.out_edges(node, data=True) and '(' in node and ')' in node:
            left_bracket_index = node.index('(')
            right_bracket_index = node.index(')')
            bracket_content = node[left_bracket_index + 1:right_bracket_index]
            if bracket_content in Mixed_G.nodes():
                for _, target, data in Mixed_G.out_edges(bracket_content, data=True):
                    if data.get('relation') == 'refers_to':
                        add_edge_with_check(Mixed_G, node, target, 'refers_to', None)
        
        # Rule for nodes with spaces
        if not Mixed_G.out_edges(node, data=True) and ' ' in node:
            if node.replace(' ', '') in Mixed_G.nodes():
                for _, target, data in Mixed_G.out_edges(node.replace(' ', ''), data=True):
                    if data.get('relation') == 'refers_to':
                        add_edge_with_check(Mixed_G, node, target, 'refers_to', None)
        
        # Rules for dataset names
        if 'dataset' not in node.lower():
            # Check for node + " dataset"
            if node + ' dataset' in Mixed_G.nodes():
                for _, target, data in Mixed_G.out_edges(node + ' dataset', data=True):
                    if data.get('relation') == 'refers_to':
                        add_edge_with_check(Mixed_G, node, target, 'refers_to', None)
            
            if node + ' Dataset' in Mixed_G.nodes():
                for _, target, data in Mixed_G.out_edges(node + ' Dataset', data=True):
                    if data.get('relation') == 'refers_to':
                        add_edge_with_check(Mixed_G, node, target, 'refers_to', None)
        
        # Rules for suffixes
        for suffix in [
            (' dataset', 8), (' Dataset', 8), 
            (' datasets', 9), (' Datasets', 9), 
            (' benchmark', 10), (' Benchmark', 10), 
            (' benchmarks', 11), (' Benchmarks', 11),
            (' corpus', 7), (' Corpus', 7),
            (' environment', 12), (' Environment', 12)
        ]:
            if node.endswith(suffix[0]):
                if node[:-suffix[1]] in Mixed_G.nodes():
                    for _, target, data in Mixed_G.out_edges(node[:-suffix[1]], data=True):
                        if data.get('relation') == 'refers_to':
                            add_edge_with_check(Mixed_G, node, target, 'refers_to', None)
        
        # Rule for comma-separated names
        if ',' in node:
            comma_index = node.index(',')
            if node[:comma_index] in Mixed_G.nodes():
                for _, target, data in Mixed_G.out_edges(node[:comma_index], data=True):
                    if data.get('relation') == 'refers_to':
                        add_edge_with_check(Mixed_G, node, target, 'refers_to', None)
        
        # Rule for case-insensitive matching
        if not Mixed_G.out_edges(node, data=True):
            if node.lower() in lower_case_nodes:
                for original_node in lower_case_nodes[node.lower()]:
                    for _, target, data in Mixed_G.out_edges(original_node, data=True):
                        if data.get('relation') == 'refers_to':
                            add_edge_with_check(Mixed_G, node, target, 'refers_to', None)
                            break
        
        # Special case for PEMS
        if not Mixed_G.out_edges(node, data=True) and node == 'https://pems.dot.ca.gov/':
            add_edge_with_check(Mixed_G, node, 'PEMS-BAY_$entity$', 'refers_to', None)
    
    return Mixed_G

def remove_uncertain_edges(G):
    """Remove edges for uncertain nodes."""
    uncertain_nodes = [
        "https://www.cs.toronto.edu/~kriz/cifar.html",
        "https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets",
        "https://storage.googleapis.com/openimages/web/index.html",
        "https://nihcc.app.box.com/v/ChestXray-NIHCC",
        "https://research.fb.com/downloads/babi/",
        "https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/",
        "https://www.csie.ntu.edu.tw/~cjlin/libsvm/",
        "5-datasets",
        "Five-dataset",
        "5-Datasets",
        "Mix datasets",
    ]

    # Delete all 'refers_to' edges from uncertain nodes
    edges_to_remove = []
    for node in uncertain_nodes:
        for _, target, data in G.out_edges(node, data=True):
            if data.get('relation') == 'refers_to':
                edges_to_remove.append((node, target))

    for node, target in edges_to_remove:
        G.remove_edge(node, target)
    
    return G

def complete_graph(Mixed_G):
    """Complete the graph by inferring missing relationships."""
    Complete_G = copy.deepcopy(Mixed_G)

    dataset_description_nodes = [
        node for node in Complete_G.nodes()
        if Complete_G.nodes[node]["type"] == "dataset_description"
    ]

    # Set of nodes that cannot have 'refers_to' edges
    S = set()

    completion_iter_max = 3
    completion_iter = 0

    print(f'Before completion, the number of refers_to edges is {len([edge for edge in Complete_G.edges(data=True) if edge[2].get("relation") == "refers_to"])}')

    while completion_iter < completion_iter_max:
        completion_iter += 1
        for node in dataset_description_nodes:
            # Get name and URL nodes
            name_node = next(
                (target for _, target, data in Complete_G.out_edges(node, data=True)
                 if data.get("relation") == "has_name"),
                None
            )
            
            url_node = next(
                (target for _, target, data in Complete_G.out_edges(node, data=True)
                 if data.get("relation") == "has_url"),
                None
            )

            # If name_node has refers_to edge and name_node not in S, add refers_to edge to url_node
            if name_node and name_node not in S and url_node and url_node not in S:
                for _, target, data in Complete_G.out_edges(name_node, data=True):
                    if data.get("relation") == "refers_to":
                        Complete_G.add_edge(url_node, target, relation="refers_to")
                        if len(list(Complete_G.out_edges(url_node, data=True))) > 1:
                            S.add(url_node)
                            Complete_G.remove_edges_from(list(Complete_G.out_edges(url_node, data=True)))

                # If url_node has refers_to edge and url_node not in S, add refers_to edge to name_node
                for _, target, data in Complete_G.out_edges(url_node, data=True):
                    if data.get("relation") == "refers_to":
                        Complete_G.add_edge(name_node, target, relation="refers_to")
                        if len(list(Complete_G.out_edges(name_node, data=True))) > 1:
                            S.add(name_node)
                            Complete_G.remove_edges_from(list(Complete_G.out_edges(name_node, data=True)))

    print(f'After completion, the number of refers_to edges is {len([edge for edge in Complete_G.edges(data=True) if edge[2].get("relation") == "refers_to"])}')
    
    return Complete_G

def finalize_refers_to_edges(Complete_G, pwc_dataset_entities):
    """Finalize refers_to edges in the completed graph."""
    # Add "refers_to" edges
    # Name
    print('Start to add name refers_to edges')
    for dataset in pwc_dataset_entities:
        add_edge_with_remove(Complete_G, dataset['name'].lower(), dataset['entity_name'], 'refers_to', dataset)
        add_edge_with_remove(Complete_G, dataset['name'].replace(' ', '-'), dataset['entity_name'], 'refers_to', dataset)
        add_edge_with_remove(Complete_G, dataset['name'].replace('-', ''), dataset['entity_name'], 'refers_to', dataset)
        add_edge_with_remove(Complete_G, dataset['name'].replace('-', ' '), dataset['entity_name'], 'refers_to', dataset)
        add_edge_with_remove(Complete_G, dataset['name'], dataset['entity_name'], 'refers_to', dataset)
    
    # Full name
    print('Start to add full_name refers_to edges')
    for dataset in pwc_dataset_entities:
        if dataset['full_name'] and dataset['full_name'] != dataset['name']:
            add_edge_with_remove(Complete_G, dataset['full_name'].lower(), dataset['entity_name'], 'refers_to', dataset)
            add_edge_with_remove(Complete_G, dataset['full_name'].replace(' ', '-'), dataset['entity_name'], 'refers_to', dataset)
            add_edge_with_remove(Complete_G, dataset['full_name'].replace('-', ''), dataset['entity_name'], 'refers_to', dataset)
            add_edge_with_remove(Complete_G, dataset['full_name'].replace('-', ' '), dataset['entity_name'], 'refers_to', dataset)
            add_edge_with_remove(Complete_G, dataset['full_name'], dataset['entity_name'], 'refers_to', dataset)
    
    # Variants
    print('Start to add variants refers_to edges')
    for dataset in pwc_dataset_entities:
        for variant in dataset['variants']:
            add_edge_with_remove(Complete_G, variant.lower(), dataset['entity_name'], 'refers_to', dataset)
            add_edge_with_remove(Complete_G, variant.replace(' ', '-'), dataset['entity_name'], 'refers_to', dataset)
            add_edge_with_remove(Complete_G, variant.replace('-', ''), dataset['entity_name'], 'refers_to', dataset)
            add_edge_with_remove(Complete_G, variant.replace('', '-'), dataset['entity_name'], 'refers_to', dataset)
            add_edge_with_remove(Complete_G, variant, dataset['entity_name'], 'refers_to', dataset)
    
    # URL
    print('Start to add url refers_to edges')
    for dataset in pwc_dataset_entities:
        if dataset['homepage']:
            add_edge_with_remove(Complete_G, dataset['homepage'], dataset['entity_name'], 'refers_to', dataset)
    
    return Complete_G

def find_entity_node(graph, dataset_description_node):
    """Find the entity node associated with a dataset description node."""
    for relation in ["has_name", "has_url"]:
        target_node = next(
            (target for _, target, data in graph.out_edges(dataset_description_node, data=True)
             if data["relation"] == relation),
            None
        )
        if target_node:
            refers_to_edge = next(
                (v for _, v, data in graph.out_edges(target_node, data=True)
                 if data["relation"] == "refers_to"),
                None
            )
            if refers_to_edge:
                return refers_to_edge
    return None

def check_dataset_descriptions(Complete_G, dataset_description_nodes):
    """Check which dataset descriptions have multiple entity references."""
    multiple_entities = []
    for node in dataset_description_nodes:
        refers_to_edges = [
            (u, v, data) for u, v, data in Complete_G.out_edges(node, data=True)
            if data["relation"] == "refers_to"
        ]
        if len(refers_to_edges) > 1:
            multiple_entities.append(node)

    print(f"\nNumber of dataset descriptions with multiple entities: {len(multiple_entities)}")
    return multiple_entities

def analyze_missing_references(Complete_G, dataset_description_nodes):
    """Analyze dataset descriptions missing entity references."""
    failed_nodes = []
    count_no_refers_to = 0
    
    for node in dataset_description_nodes:
        if not find_entity_node(Complete_G, node):
            count_no_refers_to += 1
            failed_nodes.append(node)
    
    print(f"Number of dataset description nodes without 'refers_to' edge: {count_no_refers_to}")
    print(f"Ratio of dataset description nodes without 'refers_to' edge: {count_no_refers_to / len(dataset_description_nodes):.5f}")
    
    # Decode failed nodes
    decoded_failed_nodes = []
    for item in tqdm(failed_nodes[:]):
        decoded_failed_nodes.append(ast.literal_eval(item))
    
    # Organize into DataFrame
    failed_nodes_df = pd.DataFrame(decoded_failed_nodes)
    
    return failed_nodes, failed_nodes_df

def load_test_set():
    """Load the entity resolution test set."""
    er_test_path = 'data/er_evaluation/fixed_er_test_set.json'
    er_test_set = json.load(open(er_test_path, 'r'))
    er_test_set = [x for x in er_test_set if x.get('ground truth')]
    print(f'Total number of test set examples: {len(er_test_set)}')
    
    # Simplify test set to include only necessary fields
    er_test_set = [
        {
            'arxiv id': x['arxiv id'], 
            'dataset name': x['dataset name'], 
            'ground truth': x['ground truth'], 
            'identifier': x['arxiv id'] + '_' + x['dataset name']
        } 
        for x in er_test_set
    ]
    
    return er_test_set

def make_predictions(Complete_G, filtered_data_descriptions):
    """Make entity predictions for dataset descriptions."""
    predicted_set = []
    for dataset_description in filtered_data_descriptions:
        entity_node = find_entity_node(Complete_G, dataset_description['entity_name'])
        if entity_node:
            predicted_set.append({
                'arxiv id': dataset_description['arxiv id'],
                'dataset name': dataset_description['dataset name'],
                'identifier': dataset_description['arxiv id'] + '_' + dataset_description['dataset name'],
                'predicted entity': entity_node.split('_$entity$')[0]
            })
        else:
            predicted_set.append({
                'arxiv id': dataset_description['arxiv id'],
                'dataset name': dataset_description['dataset name'],
                'identifier': dataset_description['arxiv id'] + '_' + dataset_description['dataset name'],
                'predicted entity': None
            })

    # Output statistics
    count_predicted_entity = sum(1 for x in predicted_set if x['predicted entity'])
    print(f'Number of dataset descriptions with predicted entity: {count_predicted_entity}')
    
    # Count unique predicted entities
    predicted_entities = set(x['predicted entity'] for x in predicted_set if x['predicted entity'])
    print(f'Number of unique predicted entities: {len(predicted_entities)}')
    
    # Count entities used multiple times
    predicted_entity_counts = pd.Series(x['predicted entity'] for x in predicted_set if x['predicted entity']).value_counts()
    print(f'Number of predicted entities used 2 times or more: {sum(predicted_entity_counts >= 2)}')
    
    return predicted_set

def update_test_set_predictions(er_test_set, predicted_set):
    """Update test set with predictions."""
    for test_set in er_test_set:
        for predicted in predicted_set:
            if test_set['identifier'] == predicted['identifier']:
                test_set['predicted entity'] = predicted['predicted entity']
                break
    
    return er_test_set

def main():
    """Main function to run the entity resolution pipeline."""
    # Set project directory
    set_project_dir()
    
    # Load and preprocess data
    data_descriptions = preprocess_data()
    pwc_dataset_entities = load_pwc_entities()
    
    # Build PWC graph
    PWC_G = build_pwc_graph(pwc_dataset_entities)
    
    # Create a set of unique name nodes
    unique_name_nodes = set()
    for dataset in pwc_dataset_entities:
        unique_name_nodes.add(dataset['name'])
        if dataset['full_name']:
            unique_name_nodes.add(dataset['full_name'])
    
    # Add refers_to edges
    add_refers_to_edges(PWC_G, pwc_dataset_entities, unique_name_nodes)
    
    # Filter data descriptions
    filtered_data_descriptions = filter_data_descriptions(data_descriptions)
    
    # Build mixed graph
    Mixed_G = build_mixed_graph(PWC_G, filtered_data_descriptions)
    
    # Apply rules
    Mixed_G = apply_rules(Mixed_G)
    
    # Remove uncertain edges
    Mixed_G = remove_uncertain_edges(Mixed_G)
    
    # Complete the graph
    Complete_G = complete_graph(Mixed_G)
    
    # Finalize refers_to edges
    Complete_G = finalize_refers_to_edges(Complete_G, pwc_dataset_entities)
    
    # Get dataset description nodes
    dataset_description_nodes = [
        node for node in Complete_G.nodes()
        if Complete_G.nodes[node]["type"] == "dataset_description"
    ]
    
    # Check for multiple entities
    multiple_entities = check_dataset_descriptions(Complete_G, dataset_description_nodes)
    
    # Analyze missing references
    failed_nodes, failed_nodes_df = analyze_missing_references(Complete_G, dataset_description_nodes)
    
    # Load test set
    er_test_set = load_test_set()
    
    # Make predictions
    predicted_set = make_predictions(Complete_G, filtered_data_descriptions)
    
    # Update test set with predictions
    er_test_set = update_test_set_predictions(er_test_set, predicted_set)
    
    print(f"Total filtered data descriptions: {len(filtered_data_descriptions)}")
    print(f"Total predictions: {len(predicted_set)}")
    
    # Save results if needed
    # output_file = 'data/er_evaluation/er_predictions.json'
    # with open(output_file, 'w') as f:
    #     json.dump(er_test_set, f, indent=2)
    
    return Complete_G, filtered_data_descriptions, predicted_set, er_test_set

if __name__ == "__main__":
    main() 