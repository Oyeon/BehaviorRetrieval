#!/usr/bin/env python3
"""
BehaviorRetrieval Evaluation Script - Test on all target images
Shows policy predictions for each query image
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import json
import argparse
from tqdm import tqdm

################################################################################
#                    Action Vector Generation (Your Original)
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
#                    BehaviorRetrieval Evaluator
################################################################################

class BehaviorRetrievalEvaluator:
    """BehaviorRetrieval Evaluator - direct policy inference"""
    
    def __init__(self, model_dir='./br_target_models', device='cuda'):
        self.device = device
        
        print("🔄 Loading BehaviorRetrieval models...")
        
        # Load visual encoder (same architecture as used in training)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(64, 64)
        ).to(device)
        
        # Load policy network
        self.policy = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 7)
        ).to(device)
        
        # Load trained weights
        policy_path = f"{model_dir}/policy_target_training.pth"
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"Policy not found at {policy_path}. Please train BehaviorRetrieval first.")
            
        self.policy.load_state_dict(torch.load(policy_path, map_location=device))
        
        # Load metadata if available
        metadata_path = f"{model_dir}/target_training_metadata.json"
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                print(f"✅ Training metadata loaded:")
                print(f"   - Target episodes: {self.metadata.get('target_episodes_loaded', 'N/A')}")
                print(f"   - Retrieved samples: {self.metadata.get('retrieved_samples', 'N/A')}")
                print(f"   - Total training samples: {self.metadata.get('final_policy_training_samples', 'N/A')}")
        else:
            self.metadata = {}
        
        self.visual_encoder.eval()
        self.policy.eval()
        
        print("✅ BehaviorRetrieval models loaded")
    
    def predict_action(self, image_path):
        """Predict action using trained policy"""
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to 84x84 as per BR paper
        image_resized = image.resize((84, 84), Image.LANCZOS)
        image_tensor = torch.FloatTensor(np.array(image_resized)).permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract 64-D visual features
            visual_features = self.visual_encoder(image_tensor)
            
            # Predict action with policy
            predicted_action = self.policy(visual_features).cpu().numpy()[0]
        
        return predicted_action, visual_features.cpu().numpy()[0]

def evaluate_all_br_samples(args):
    """Evaluate BehaviorRetrieval on all target images"""
    
    # Initialize evaluator
    evaluator = BehaviorRetrievalEvaluator(model_dir=args.model_dir, device=args.device)
    
    all_results = []
    
    print("\n" + "="*80)
    print("🔍 BEHAVIOR RETRIEVAL EVALUATION ON ALL TARGET SAMPLES")
    print("="*80)
    print(f"📁 Target directory: {args.target_dir}")
    print(f"💾 Model directory: {args.model_dir}")
    print(f"🧠 Policy type: Direct neural network inference (no retrieval during evaluation)")
    print("="*80)
    
    # Iterate through all episodes
    for episode_id in range(1, 28):  # Episodes 1-27
        episode_path = Path(args.target_dir) / str(episode_id)
        if not episode_path.exists():
            continue
        
        print(f"\n📂 Episode {episode_id}:")
        episode_results = []
        
        for step in range(1, 18):  # Steps 1-17
            # Try different image formats
            img_file = episode_path / f"{step:02d}.jpg"
            if not img_file.exists():
                img_file = episode_path / f"{step}.jpg"
            
            if not img_file.exists():
                continue
            
            # Get ground truth action
            gt_action_3d = get_action_vector(step, str(episode_id))
            gt_action_7d = np.array(gt_action_3d + [0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            
            # Get BR prediction
            predicted_action, visual_features = evaluator.predict_action(str(img_file))
            
            # Calculate error
            error = np.linalg.norm(predicted_action - gt_action_7d)
            
            # Calculate per-dimension errors
            position_error = np.linalg.norm(predicted_action[:3] - gt_action_7d[:3])
            rotation_error = np.linalg.norm(predicted_action[3:6] - gt_action_7d[3:6])
            gripper_error = abs(predicted_action[6] - gt_action_7d[6])
            
            # Store result
            result = {
                'episode': episode_id,
                'step': step,
                'image_path': str(img_file),
                'ground_truth_action': gt_action_7d.tolist(),
                'predicted_action': predicted_action.tolist(),
                'error': float(error),
                'position_error': float(position_error),
                'rotation_error': float(rotation_error),
                'gripper_error': float(gripper_error),
                'visual_features_norm': float(np.linalg.norm(visual_features))
            }
            
            episode_results.append(result)
            
            # Print summary
            if args.verbose:
                print(f"\n  Step {step:2d}:")
                print(f"    Ground truth: {gt_action_7d}")
                print(f"    Predicted:    {predicted_action}")
                print(f"    Total error: {error:.6f}")
                print(f"    Position error: {position_error:.6f}")
                print(f"    Rotation error: {rotation_error:.6f}")
                print(f"    Gripper error: {gripper_error:.6f}")
            else:
                # Compact output
                print(f"  Step {step:2d}: error={error:.4f} "
                      f"(pos={position_error:.4f}, rot={rotation_error:.4f}, grip={gripper_error:.4f})")
        
        all_results.extend(episode_results)
        
        # Episode summary
        if episode_results:
            avg_error = np.mean([r['error'] for r in episode_results])
            avg_pos_error = np.mean([r['position_error'] for r in episode_results])
            avg_rot_error = np.mean([r['rotation_error'] for r in episode_results])
            print(f"  📊 Episode {episode_id} average - Total: {avg_error:.4f}, "
                  f"Pos: {avg_pos_error:.4f}, Rot: {avg_rot_error:.4f}")
    
    # Overall summary
    print("\n" + "="*80)
    print("📊 OVERALL BEHAVIOR RETRIEVAL RESULTS")
    print("="*80)
    
    all_errors = [r['error'] for r in all_results]
    all_pos_errors = [r['position_error'] for r in all_results]
    all_rot_errors = [r['rotation_error'] for r in all_results]
    all_grip_errors = [r['gripper_error'] for r in all_results]
    
    print(f"Total samples evaluated: {len(all_results)}")
    print(f"\nOverall Error Statistics:")
    print(f"  Mean total error: {np.mean(all_errors):.6f}")
    print(f"  Std total error: {np.std(all_errors):.6f}")
    print(f"  Min total error: {np.min(all_errors):.6f}")
    print(f"  Max total error: {np.max(all_errors):.6f}")
    
    print(f"\nPer-Component Error Statistics:")
    print(f"  Position - Mean: {np.mean(all_pos_errors):.6f}, Std: {np.std(all_pos_errors):.6f}")
    print(f"  Rotation - Mean: {np.mean(all_rot_errors):.6f}, Std: {np.std(all_rot_errors):.6f}")
    print(f"  Gripper  - Mean: {np.mean(all_grip_errors):.6f}, Std: {np.std(all_grip_errors):.6f}")
    
    # Find best and worst predictions
    best = min(all_results, key=lambda x: x['error'])
    worst = max(all_results, key=lambda x: x['error'])
    
    print(f"\n🏆 Best prediction: Episode {best['episode']}, Step {best['step']} (error={best['error']:.6f})")
    print(f"❌ Worst prediction: Episode {worst['episode']}, Step {worst['step']} (error={worst['error']:.6f})")
    
    # Analyze error patterns
    print("\n📈 Error Pattern Analysis:")
    
    # Group errors by episode
    episode_errors = {}
    for result in all_results:
        ep = result['episode']
        if ep not in episode_errors:
            episode_errors[ep] = []
        episode_errors[ep].append(result['error'])
    
    # Find best and worst episodes
    episode_avg_errors = {ep: np.mean(errors) for ep, errors in episode_errors.items()}
    best_episodes = sorted(episode_avg_errors.items(), key=lambda x: x[1])[:3]
    worst_episodes = sorted(episode_avg_errors.items(), key=lambda x: x[1], reverse=True)[:3]
    
    print("Best performing episodes:")
    for ep, err in best_episodes:
        print(f"  Episode {ep}: avg error = {err:.6f}")
    
    print("Worst performing episodes:")
    for ep, err in worst_episodes:
        print(f"  Episode {ep}: avg error = {err:.6f}")
    
    # Save detailed results
    output_file = args.output_file
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Create summary report
    summary = {
        'total_samples': len(all_results),
        'model_metadata': evaluator.metadata,
        'overall_statistics': {
            'mean_error': float(np.mean(all_errors)),
            'std_error': float(np.std(all_errors)),
            'min_error': float(np.min(all_errors)),
            'max_error': float(np.max(all_errors))
        },
        'component_statistics': {
            'position': {
                'mean': float(np.mean(all_pos_errors)),
                'std': float(np.std(all_pos_errors))
            },
            'rotation': {
                'mean': float(np.mean(all_rot_errors)),
                'std': float(np.std(all_rot_errors))
            },
            'gripper': {
                'mean': float(np.mean(all_grip_errors)),
                'std': float(np.std(all_grip_errors))
            }
        },
        'best_prediction': {
            'episode': best['episode'],
            'step': best['step'],
            'error': best['error']
        },
        'worst_prediction': {
            'episode': worst['episode'],
            'step': worst['step'],
            'error': worst['error']
        },
        'episode_performance': episode_avg_errors
    }
    
    summary_file = output_file.replace('.json', '_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📊 Summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate BehaviorRetrieval on all target samples')
    
    # Paths
    parser.add_argument('--target_dir', type=str, 
                        default='/mnt/storage/owen/robot-dataset/rt-cache/raw/',
                        help='Path to target dataset')
    parser.add_argument('--model_dir', type=str, default='./br_target_models',
                        help='Path to BehaviorRetrieval models')
    
    # Parameters
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed results for each step')
    parser.add_argument('--output_file', type=str, default='br_evaluation_results.json',
                        help='Output file for detailed results')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.target_dir):
        print(f"❌ Target directory not found: {args.target_dir}")
        return
    
    if not os.path.exists(args.model_dir):
        print(f"❌ Model directory not found: {args.model_dir}")
        print("Please train BehaviorRetrieval first using: python behavior_retrieval_target_training.py")
        return
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    # Run evaluation
    evaluate_all_br_samples(args)

if __name__ == '__main__':
    main()