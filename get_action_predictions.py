#!/usr/bin/env python3
"""
Simple script to get only action predictions from BehaviorRetrieval
Returns clean action vectors without extra information
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import argparse

################################################################################
#                    BehaviorRetrieval Implementation (Minimal)
################################################################################

class BRPredictor:
    def __init__(self, model_dir='./br_target_models', device='cuda'):
        self.device = device
        
        print("🔄 Loading BehaviorRetrieval models...")
        
        # Load networks
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
        
        # Load weights
        policy_path = f"{model_dir}/policy_target_training.pth"
        if not os.path.exists(policy_path):
            print(f"⚠️  BR policy not found at {policy_path}")
            self.initialized = False
            return
            
        self.policy.load_state_dict(torch.load(policy_path, map_location=device))
        
        self.visual_encoder.eval()
        self.policy.eval()
        self.initialized = True
        print("✅ BehaviorRetrieval models loaded")
    
    def predict(self, image_path):
        """Get action prediction only"""
        if not self.initialized:
            return None
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_resized = image.resize((84, 84), Image.LANCZOS)
        image_tensor = torch.FloatTensor(np.array(image_resized)).permute(2, 0, 1) / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            visual_features = self.visual_encoder(image_tensor)
            predicted_action = self.policy(visual_features).cpu().numpy()[0]
        
        return predicted_action

################################################################################
#                    Main Functions
################################################################################

def get_predictions(image_path, model_dir='./br_target_models', device='cuda'):
    """Get action predictions from BehaviorRetrieval"""
    
    # Initialize predictor
    br = BRPredictor(model_dir, device)
    
    # Get prediction
    br_action = br.predict(image_path) if br.initialized else None
    
    return br_action

def main():
    parser = argparse.ArgumentParser(description='Get BehaviorRetrieval action predictions only')
    parser.add_argument('image_path', type=str, help='Path to query image')
    parser.add_argument('--model_dir', type=str, default='./br_target_models')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--format', choices=['array', 'list', 'csv'], default='array',
                        help='Output format')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.image_path):
        print(f"Error: Image not found: {args.image_path}")
        return
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    # Get prediction
    br_action = get_predictions(args.image_path, args.model_dir, args.device)
    
    # Format output
    if args.format == 'array':
        if br_action is not None:
            print("BR:", br_action)
        else:
            print("BR: Model not loaded")
    
    elif args.format == 'list':
        if br_action is not None:
            print("BR:", br_action.tolist())
        else:
            print("BR: Model not loaded")
    
    elif args.format == 'csv':
        if br_action is not None:
            print("BR," + ",".join(map(str, br_action)))
        else:
            print("BR,Model not loaded")

if __name__ == '__main__':
    main()