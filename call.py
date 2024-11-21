import subprocess
import json
import os
import re
import shutil
from datetime import datetime

def run_training(script_path, checkpoints_dir, output_dir, run_id):
    """
    Run the training script, capture the best validation Dice score,
    and copy the corresponding checkpoint file.
    """
    try:
        # Run the training script and capture the output
        process = subprocess.Popen(
            ['python3', script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        stdout, stderr = process.communicate()
        
        # Check if there were errors
        if process.returncode != 0:
            raise RuntimeError(f"Training run {run_id} failed with error:\n{stderr}")
        
        # Extract the best Dice score and epoch from stdout
        best_score = None
        best_epoch = None
        for line in stdout.splitlines():
            # Look for validation Dice score
            match_score = re.search(r'Validation Dice score:\s*([\d.]+)', line)
            if match_score:
                best_score = float(match_score.group(1))
            
            # Look for checkpoint saved messages
            match_epoch = re.search(r'Checkpoint (\d+) saved!', line)
            if match_epoch:
                best_epoch = int(match_epoch.group(1))
        
        if best_score is None or best_epoch is None:
            raise ValueError(f"Could not extract Dice score or epoch for run {run_id}. Check the output:\n{stdout}")
        
        # Format the new filename
        timestamp = datetime.now().strftime('%m%d%H%M')
        new_filename = f"{timestamp}_score{best_score:.4f}.pth"

        # Copy the checkpoint file to the record directory
        source_file = os.path.join(checkpoints_dir, f"checkpoint_epoch{best_epoch}.pth")
        destination_file = os.path.join(output_dir, new_filename)
        os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
        shutil.copy(source_file, destination_file)
        print(f"Run {run_id}: Best checkpoint copied to {destination_file}")

        return best_score

    except Exception as e:
        print(f"Error in run {run_id}: {e}")
        return None

def main():
    script_path = 'train.py'  # Path to the training script
    checkpoints_dir = './checkpoints'  # Directory where checkpoints are saved
    output_dir = './checkpoints/record'  # Directory to save best checkpoints
    output_file = 'training_results.json'  # Output JSON file
    runs = 5  # Number of times to run the training script
    
    results = {}
    
    for i in range(1, runs + 1):
        print(f"Starting training run {i}/{runs}...")
        best_score = run_training(script_path, checkpoints_dir, output_dir, i)
        if best_score is not None:
            results[f"Run_{i}"] = best_score
    
    # Write the results to a JSON file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
