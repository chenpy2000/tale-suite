"""
Parse diagnostic evaluation results from terminal output.

Usage:
    python parse_diagnostic_results.py --input vqvae_diagnostic.txt --output vqvae_results.json
"""

import re
import json
import argparse
from pathlib import Path


def parse_terminal_output(text):
    """
    Parse terminal output to extract per-environment results.
    Handles interrupted/restarted runs by combining all environment results.
    
    Returns:
        dict: {
            'agent': str,
            'results': [{'env': str, 'score': float, 'max_score': int, 
                         'time': float, 'doom_loops': int}, ...],
            'mean_score': float,
            'total_time': float,
            'total_doom_loops': int
        }
    """
    # Extract agent name
    agent_match = re.search(r'tales_(\w+)_', text)
    agent_name = agent_match.group(1) if agent_match else 'unknown'
    
    # Parse individual environment results
    # Flexible pattern: handles spaces inside percentage parentheses like "( 0.00%)"
    pattern = r'(\w+)\s+Steps:\s+\d+/\s*\d+\s+Time:\s+(\d+):(\d+):(\d+)\s+\d+\s+resets\s+Score:\s+(\d+)/\s*(\d+)\s+\(\s*[\d.]+%\)\s+TokenEff:\s+[\d.\-]+\s+DoomLoop:\s+(\d+)'
    
    results = []
    env_names_seen = set()
    
    for match in re.finditer(pattern, text):
        env_name = match.group(1)
        
        # Skip duplicates (keep first occurrence for each environment)
        if env_name in env_names_seen:
            continue
        env_names_seen.add(env_name)
        
        hours = int(match.group(2))
        minutes = int(match.group(3))
        seconds = int(match.group(4))
        time_seconds = hours * 3600 + minutes * 60 + seconds
        
        score = int(match.group(5))
        max_score = int(match.group(6))
        doom_loops = int(match.group(7))
        
        results.append({
            'env': env_name,
            'score': score,
            'max_score': max_score,
            'normalized_score': score / max_score if max_score > 0 else 0.0,
            'time_seconds': time_seconds,
            'doom_loops': doom_loops
        })
    
    # Calculate summary statistics from parsed results
    if results:
        total_score = sum(r['score'] for r in results)
        total_max_score = sum(r['max_score'] for r in results)
        mean_score = (total_score / total_max_score) if total_max_score > 0 else 0.0
        
        total_time = sum(r['time_seconds'] for r in results)
        total_doom_loops = sum(r['doom_loops'] for r in results)
    else:
        mean_score = 0.0
        total_time = 0.0
        total_doom_loops = 0
    
    return {
        'agent': agent_name,
        'results': results,
        'mean_score': mean_score,
        'total_time_seconds': total_time,
        'total_time_minutes': total_time / 60.0,
        'total_time_hours': total_time / 3600.0,
        'total_doom_loops': total_doom_loops,
        'num_environments': len(results)
    }


def load_skill_categories(json_path):
    """Load skill category mappings from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def categorize_by_skill(parsed_data, skill_categories):
    """
    Categorize results by reasoning skill.
    
    Returns:
        dict: {skill: {'scores': [...], 'mean': float}}
    """
    # Create env -> skill mapping
    env_to_skill = {}
    for skill, tasks in skill_categories.items():
        for task_info in tasks:
            env_to_skill[task_info['task']] = skill
    
    # Group results by skill
    skill_results = {
        'spatial': [],
        'deductive': [],
        'inductive': [],
        'grounded': []
    }
    
    for result in parsed_data['results']:
        env = result['env']
        skill = env_to_skill.get(env)
        if skill:
            skill_results[skill].append(result['normalized_score'])
    
    # Calculate mean per skill
    skill_scores = {}
    for skill, scores in skill_results.items():
        skill_scores[skill] = {
            'scores': scores,
            'mean': sum(scores) / len(scores) if scores else 0.0,
            'count': len(scores)
        }
    
    return skill_scores


def main():
    parser = argparse.ArgumentParser(description='Parse diagnostic evaluation results')
    parser.add_argument('--input', required=True, help='Input terminal output file')
    parser.add_argument('--output', required=True, help='Output JSON file')
    parser.add_argument('--skills', default='data/diagnostic_tasks.json',
                       help='Path to skill categories JSON')
    args = parser.parse_args()
    
    # Read input file
    with open(args.input, 'r') as f:
        text = f.read()
    
    # Parse results
    parsed = parse_terminal_output(text)
    
    # Load skill categories and categorize results
    if Path(args.skills).exists():
        skills = load_skill_categories(args.skills)
        parsed['skill_scores'] = categorize_by_skill(parsed, skills)
    
    # Save to JSON
    with open(args.output, 'w') as f:
        json.dump(parsed, f, indent=2)
    
    # Print summary
    print(f"Parsed results for agent: {parsed['agent']}")
    print(f"Environments tested: {parsed['num_environments']}")
    print(f"Mean score: {parsed['mean_score']*100:.2f}%")
    print(f"Total time: {parsed['total_time_minutes']:.2f} minutes")
    print(f"Total doom loops: {parsed['total_doom_loops']}")
    
    if 'skill_scores' in parsed:
        print("\nPer-skill scores:")
        for skill, data in parsed['skill_scores'].items():
            print(f"  {skill}: {data['mean']*100:.2f}% ({data['count']} tasks)")
    
    print(f"\nSaved to: {args.output}")


if __name__ == '__main__':
    main()
