import json
import os
from collections import OrderedDict

def calculate_and_save_scores(input_path):
    # Build the complete path to the math folder
    base_path = os.path.join(input_path, "math")
    scores = []
    total_score = 0
    file_count = 0
    
    # Check if math folder exists
    if not os.path.exists(base_path):
        print(f"Error: {base_path} does not exist!")
        return
    
    # Traverse and collect all scores
    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path, subdir)
        if os.path.isdir(subdir_path):
            metric_file = os.path.join(subdir_path, "metrics.json")
            
            if os.path.exists(metric_file):
                with open(metric_file, 'r') as f:
                    data = json.load(f)
                    
                    # Get complete score string and value
                    metrics = data["metrics"][0]  # Get the first metrics object
                    task_name = metrics["task"]
                    score = metrics["exact_match_flex"]
                    
                    # Save task name and score
                    scores.append({
                        "task": task_name,
                        "score": score
                    })
                    
                    total_score += score
                    file_count += 1
    
    # Calculate average score
    average_score = total_score / file_count if file_count > 0 else 0
    
    # Create final results
    results = []
    
    # Add scores for each task
    for score_info in scores:
        result = OrderedDict([
            ("task", score_info["task"]),
            ("score", score_info["score"])
        ])
        results.append(result)
    
    # Add average score
    average_result = OrderedDict([
        ("task", "average"),
        ("score", round(average_score, 6))
    ])
    results.append(average_result)
    
    # Save to math folder
    output_path = os.path.join(base_path, 'task_scores_flex.jsonl')
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"Results have been saved to {output_path}")
    print(f"\nSummary:")
    print(f"Number of tasks processed: {file_count}")
    print(f"Average score: {average_score:.6f}")

if __name__ == "__main__":
    # Get input path from command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Calculate average scores from metrics.json files')
    parser.add_argument('input_path', type=str, help='Path to the directory containing math folder')
    args = parser.parse_args()
    
    calculate_and_save_scores(args.input_path)