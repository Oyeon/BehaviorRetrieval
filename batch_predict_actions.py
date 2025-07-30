#!/usr/bin/env python3
"""
Batch prediction script for BehaviorRetrieval - get action vectors for all samples
Returns only action predictions in clean format
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import argparse
from pathlib import Path
import csv
import json
from tqdm import tqdm

################################################################################
#                    Action Vector Generation (Ground Truth)
################################################################################

def get_action_vector(i: int, epi: str):
    """Your original Franka canned plan action vector generation"""
    def f(a,b,c,d,e): return \
        a if 1<=i<=5 else b if 6<=i<=8 else c if 9<=i<=12 else d if 13<=i<=17 else e
    _L1 = f([0, 0.035,0], [0,0,-0.055], [0,-0.02,0],  [0,0,-0.055], [0,0,0])
    _R1 = f([0,-0.035,0], [0,0,-0.055], [0, 0.02,0],  [0,0,-0.055], [0,0,0])
    _F1 = f([0.01,0,0],  [0,0,-0.055], [0,0.01,0],  [0,0,-0.055], [0,0,0])

    _L2 = f([0, 0.035,0], [0,0,-0.045], [-0.01, 0,0],  [0,0,-0.045], [0,0,0])
    _R2 = f([0,-0.035,0], [0,0,-0.045], [-0.01, 0,0],  [0,0,-0.045], [0,0,0])
    _F2 = f([0.02,0,0],  [0,0,-0.045], [-0.01, 0,0],  [0,0,-0.045], [0,0,0])
    
    _L3 = f([0, 0.035,0], [0,0,-0.055], [0, 0.01,0],  [0,0,-0.055], [0,0,0])
    _R3 = f([0,-0.035,0], [0,0,-0.055], [0, -0.01,0],  [0,0,-0.055], [0,0,0])
    _F3 = f([0.01,0,0],  [0,0,-0.055], [-0.01,0,0],  [0,0,-0.055], [0,0,0])

    families  = [[_L1,_L2,_L3], [_R1,_R2,_R3], [_F1,_F2,_F3]]

    try:
        eid = int(epi)
    except ValueError:
        return [0,0,0]
    if not 1<=eid<=28: return [0,0,0]
    fam  = (eid-1) % 3
    var  = ((eid-1)//3) % 3
    return families[fam][var]

################################################################################
#                    BehaviorRetrieval Predictor
################################################################################

class BRPredictor:
    def __init__(self, model_dir='./br_target_models', device='cuda'):
        self.device = device
        
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64)
        ).to(device)
        
        self.policy = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        ).to(device)
        
        policy_path = f"{model_dir}/policy_target_training.pth"
        if not os.path.exists(policy_path):
            print(f"⚠️  BR policy not found at {policy_path}")
            self.initialized = False
            return
            
        self.policy.load_state_dict(torch.load(policy_path, map_location=device))
        
        self.visual_encoder.eval()
        self.policy.eval()
        self.initialized = True
    
    def predict(self, image_path):
        if not self.initialized:
            return None
        
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((84, 84), Image.LANCZOS)
        image_tensor = torch.FloatTensor(np.array(image_resized)).permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            visual_features = self.visual_encoder(image_tensor)
            predicted_action = self.policy(visual_features).cpu().numpy()[0]
        
        return predicted_action

################################################################################
#                    Batch Processing
################################################################################

def batch_predict(target_dir, output_format='console', output_file=None, 
                 model_dir='./br_target_models', device='cuda'):
    """Batch predict actions for all images in target directory"""
    
    # Initialize predictor
    print("🔄 Loading BehaviorRetrieval model...")
    br = BRPredictor(model_dir, device)
    
    if not br.initialized:
        print("❌ BehaviorRetrieval model not loaded")
        return
    
    print("✅ BehaviorRetrieval model loaded")
    
    # Collect all image paths
    image_paths = []
    for episode_id in range(1, 28):
        episode_path = Path(target_dir) / str(episode_id)
        if not episode_path.exists():
            continue
        
        for step in range(1, 18):
            img_file = episode_path / f"{step:02d}.jpg"
            if not img_file.exists():
                img_file = episode_path / f"{step}.jpg"
            
            if img_file.exists():
                image_paths.append((episode_id, step, str(img_file)))
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process images
    results = []
    
    for episode_id, step, img_path in tqdm(image_paths, desc="Processing"):
        # Get ground truth
        gt_action_3d = get_action_vector(step, str(episode_id))
        gt_action_7d = np.array(gt_action_3d + [0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # Get BR prediction
        br_action = br.predict(img_path)
        
        result = {
            'episode': episode_id,
            'step': step,
            'image_path': img_path,
            'ground_truth': gt_action_7d.tolist(),
            'br_action': br_action.tolist() if br_action is not None else None
        }
        
        # Calculate error if prediction exists
        if br_action is not None:
            error = np.linalg.norm(br_action - gt_action_7d)
            result['error'] = float(error)
        
        results.append(result)
        
        # Console output
        if output_format == 'console':
            if br_action is not None:
                br_str = "[" + ", ".join(f"{x:.4f}" for x in br_action) + "]"
                gt_str = "[" + ", ".join(f"{x:.4f}" for x in gt_action_7d) + "]"
                error_str = f"error: {result.get('error', 0):.4f}" if 'error' in result else "error: N/A"
                print(f"Ep{episode_id:2d}/Step{step:2d}: BR: {br_str} | GT: {gt_str} | {error_str}")
            else:
                print(f"Ep{episode_id:2d}/Step{step:2d}: BR: Failed to predict")
    
    # Save to file if requested
    if output_file:
        if output_format == 'json':
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {output_file}")
        
        elif output_format == 'csv':
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                header = ['episode', 'step', 'image_path', 'error']
                header.extend([f'gt_action_{i}' for i in range(7)])
                header.extend([f'br_action_{i}' for i in range(7)])
                writer.writerow(header)
                
                # Data
                for result in results:
                    row = [result['episode'], result['step'], result['image_path'], 
                           result.get('error', 'N/A')]
                    row.extend(result['ground_truth'])
                    if result['br_action']:
                        row.extend(result['br_action'])
                    else:
                        row.extend([None] * 7)
                    writer.writerow(row)
            print(f"Results saved to {output_file}")
    
    # Print summary
    valid_results = [r for r in results if r.get('error') is not None]
    if valid_results:
        errors = [r['error'] for r in valid_results]
        print(f"\n📊 Summary:")
        print(f"  Total samples: {len(results)}")
        print(f"  Valid predictions: {len(valid_results)}")
        print(f"  Mean error: {np.mean(errors):.6f}")
        print(f"  Std error: {np.std(errors):.6f}")
        print(f"  Min error: {np.min(errors):.6f}")
        print(f"  Max error: {np.max(errors):.6f}")

def main():
    parser = argparse.ArgumentParser(description='Batch predict BehaviorRetrieval actions for all samples')
    
    parser.add_argument('--target_dir', type=str, 
                        default='/mnt/storage/owen/robot-dataset/rt-cache/raw/',
                        help='Path to target dataset')
    parser.add_argument('--model_dir', type=str, default='./br_target_models')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--format', choices=['console', 'json', 'csv'], default='console',
                        help='Output format')
    parser.add_argument('--output', type=str, help='Output file (required for json/csv)')
    
    args = parser.parse_args()
    
    if args.format in ['json', 'csv'] and not args.output:
        print("Error: --output required for json/csv format")
        return
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    # Run batch prediction
    batch_predict(
        args.target_dir, 
        args.format, 
        args.output,
        args.model_dir,
        args.device
    )

if __name__ == '__main__':
    main()