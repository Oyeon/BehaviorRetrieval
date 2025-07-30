#!/usr/bin/env python3
"""
Test BehaviorRetrieval on a single image sample
Shows policy prediction for a single query image
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import argparse

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
#                    BehaviorRetrieval Implementation
################################################################################

class BehaviorRetrievalEvaluator:
    """BehaviorRetrieval Evaluator - direct policy inference"""
    
    def __init__(self, model_dir='./br_target_models', device='cuda'):
        self.device = device
        self.model_dir = model_dir
        
        print("🔄 Loading BehaviorRetrieval models...")
        
        # Load visual encoder
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
            print(f"⚠️  BR policy not found at {policy_path}")
            self.initialized = False
            return
            
        self.policy.load_state_dict(torch.load(policy_path, map_location=device))
        
        # Load metadata if available
        metadata_path = f"{model_dir}/target_training_metadata.json"
        if os.path.exists(metadata_path):
            import json
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
        self.initialized = True
        print("✅ BehaviorRetrieval models loaded")
    
    def predict_action(self, image_path):
        """Predict action using trained policy"""
        if not self.initialized:
            return None, None
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Resize to 84x84 as per BR paper
        image_resized = image.resize((84, 84), Image.LANCZOS)
        image_tensor = torch.FloatTensor(np.array(image_resized)).permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Extract visual features
            visual_features = self.visual_encoder(image_tensor)
            
            # Predict action
            predicted_action = self.policy(visual_features).cpu().numpy()[0]
        
        return predicted_action, visual_features.cpu().numpy()[0]

################################################################################
#                    Main Test Function
################################################################################

def test_single_sample(image_path, episode=None, step=None, device='cuda'):
    """Test BehaviorRetrieval on a single sample"""
    
    print("="*80)
    print(f"🧠 TESTING BEHAVIOR RETRIEVAL ON: {image_path}")
    print("="*80)
    
    # Initialize evaluator
    br_eval = BehaviorRetrievalEvaluator(
        model_dir='./br_target_models', 
        device=device
    )
    
    # Get ground truth if episode and step are provided
    gt_action = None
    if episode is not None and step is not None:
        gt_action_3d = get_action_vector(step, str(episode))
        gt_action = np.array(gt_action_3d + [0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        print(f"📍 Ground truth action: {gt_action}")
    
    print("\n" + "-"*80)
    print("🧠 BEHAVIOR RETRIEVAL RESULTS")
    print("-"*80)
    
    if br_eval.initialized:
        br_pred, br_features = br_eval.predict_action(image_path)
        
        print(f"🎯 BR prediction: {br_pred}")
        
        if gt_action is not None:
            br_error = np.linalg.norm(br_pred - gt_action)
            print(f"❌ BR error: {br_error:.6f}")
            
            # Calculate per-component errors
            position_error = np.linalg.norm(br_pred[:3] - gt_action[:3])
            rotation_error = np.linalg.norm(br_pred[3:6] - gt_action[3:6])
            gripper_error = abs(br_pred[6] - gt_action[6])
            
            print(f"📏 Component errors:")
            print(f"   Position error: {position_error:.6f}")
            print(f"   Rotation error: {rotation_error:.6f}")
            print(f"   Gripper error: {gripper_error:.6f}")
        
        print(f"🔢 Feature vector norm: {np.linalg.norm(br_features):.6f}")
        
        # Show training metadata if available
        if br_eval.metadata:
            print(f"\n📊 Model Training Info:")
            for key, value in br_eval.metadata.items():
                print(f"   {key}: {value}")
    else:
        print("❌ BehaviorRetrieval not initialized")

def main():
    parser = argparse.ArgumentParser(description='Test BehaviorRetrieval on a single sample')
    
    parser.add_argument('image_path', type=str, help='Path to the query image')
    parser.add_argument('--episode', type=int, help='Episode number (for ground truth)')
    parser.add_argument('--step', type=int, help='Step number (for ground truth)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"❌ Image not found: {args.image_path}")
        return
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    # Run test
    test_single_sample(
        args.image_path, 
        args.episode, 
        args.step, 
        args.device
    )

if __name__ == '__main__':
    main()