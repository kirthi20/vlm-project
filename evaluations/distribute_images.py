#!/usr/bin/env python3
"""
GPU Image Processing Distributor

Distributes image processing across multiple GPU instances with configurable
image ranges and GPU assignments.

Usage:
    python distribute_images.py --total-images 5000 --instances 2 --gpus 0,1
    python distribute_images.py --total-images 5000 --instances 4 --gpus 0,0,1,1
"""

import argparse
import subprocess
import sys
from typing import List, Tuple
import math


def calculate_image_ranges(total_images: int, num_instances: int) -> List[Tuple[int, int]]:
    """Calculate start and end indices for each instance."""
    images_per_instance = math.ceil(total_images / num_instances)
    ranges = []
    
    for i in range(num_instances):
        start_idx = i * images_per_instance
        end_idx = min((i + 1) * images_per_instance, total_images)
        
        # Skip if this instance would have no images
        if start_idx >= total_images:
            break
            
        ranges.append((start_idx, end_idx))
    
    return ranges


def parse_gpu_assignment(gpu_string: str, num_instances: int) -> List[int]:
    """Parse GPU assignment string and validate against number of instances."""
    if not gpu_string:
        # Default: cycle through available GPUs
        return [i % 4 for i in range(num_instances)]
    
    gpus = [int(x.strip()) for x in gpu_string.split(',')]
    
    if len(gpus) == 1:
        # Single GPU specified, use for all instances
        return [gpus[0]] * num_instances
    elif len(gpus) == num_instances:
        # Exact mapping provided
        return gpus
    else:
        raise ValueError(f"GPU assignment length ({len(gpus)}) doesn't match instances ({num_instances})")


def run_instance(gpu_id: int, start_idx: int, end_idx: int, script_path: str, extra_args: List[str]) -> subprocess.Popen:
    """Launch a single instance on specified GPU with image range."""
    env_vars = f"CUDA_VISIBLE_DEVICES={gpu_id}"
    
    # Build command
    cmd = [
        "env", f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "python", script_path,
        "--start-idx", str(start_idx),
        "--end-idx", str(end_idx),
        "--gpu-id", str(gpu_id)
    ] + extra_args
    
    print(f"Starting instance on GPU {gpu_id}: images {start_idx}-{end_idx-1}")
    print(f"Command: {' '.join(cmd[1:])}")  # Skip 'env' for cleaner output
    
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser(
        description="Distribute image processing across multiple GPU instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2 instances on GPUs 0 and 1
  python distribute_images.py --total-images 5000 --instances 2 --gpus 0,1 your_script.py

  # 4 instances, 2 on each GPU
  python distribute_images.py --total-images 5000 --instances 4 --gpus 0,0,1,1 your_script.py

  # 3 instances all on GPU 0
  python distribute_images.py --total-images 5000 --instances 3 --gpus 0 your_script.py

  # Pass additional arguments to your script
  python distribute_images.py --total-images 5000 --instances 2 --gpus 0,1 your_script.py --batch-size 32 --model resnet50
        """
    )
    
    parser.add_argument("script", help="Path to your image processing script")
    parser.add_argument("--total-images", type=int, required=True, 
                       help="Total number of images to process")
    parser.add_argument("--instances", type=int, default=2, choices=range(1, 5),
                       help="Number of instances to run (1-4)")
    parser.add_argument("--gpus", type=str, default="",
                       help="GPU assignment (e.g., '0,1' or '0,0,1,1'). If not specified, cycles through 0-3")
    parser.add_argument("--wait", action="store_true", default=True,
                       help="Wait for all instances to complete (default: True)")
    parser.add_argument("--no-wait", dest="wait", action="store_false",
                       help="Don't wait for instances to complete")
    
    # Parse known args to allow passing extra arguments to the target script
    args, extra_args = parser.parse_known_args()
    
    try:
        # Calculate image ranges
        ranges = calculate_image_ranges(args.total_images, args.instances)
        
        # Parse GPU assignments
        gpu_assignments = parse_gpu_assignment(args.gpus, len(ranges))
        
        print(f"Distributing {args.total_images} images across {len(ranges)} instances:")
        for i, ((start, end), gpu) in enumerate(zip(ranges, gpu_assignments)):
            print(f"  Instance {i+1}: GPU {gpu}, images {start}-{end-1} ({end-start} images)")
        print()
        
        # Launch all instances
        processes = []
        for (start_idx, end_idx), gpu_id in zip(ranges, gpu_assignments):
            proc = run_instance(gpu_id, start_idx, end_idx, args.script, extra_args)
            processes.append(proc)
        
        if args.wait:
            print(f"\nWaiting for {len(processes)} instances to complete...")
            
            # Wait for all processes and collect results
            success_count = 0
            for i, proc in enumerate(processes):
                stdout, stderr = proc.communicate()
                
                if proc.returncode == 0:
                    success_count += 1
                    print(f"✓ Instance {i+1} completed successfully")
                else:
                    print(f"✗ Instance {i+1} failed with return code {proc.returncode}")
                    if stderr:
                        print(f"  Error: {stderr.decode().strip()}")
            
            print(f"\nCompleted: {success_count}/{len(processes)} instances successful")
            
            # Exit with error if any instance failed
            if success_count < len(processes):
                sys.exit(1)
        else:
            print(f"\nLaunched {len(processes)} instances. Not waiting for completion.")
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()