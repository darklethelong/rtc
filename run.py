#!/usr/bin/env python
import os
import argparse
import subprocess
import sys

# Force CPU usage to avoid CUDA errors
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def check_requirements():
    """Check if all requirements are installed."""
    try:
        import tensorflow as tf
        import streamlit as st
        import pandas as pd
        import numpy as np
        import plotly.graph_objects as go
        import nltk
        return True
    except ImportError as e:
        print(f"Missing required dependency: {e}")
        return False

def install_requirements():
    """Install all requirements from requirements.txt."""
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')
    
    if os.path.exists(req_path):
        print("Installing required dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
        return True
    else:
        print("requirements.txt not found!")
        return False

def train_model(args):
    """Run the training script with the given arguments."""
    train_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'train.py')
    
    cmd = [
        sys.executable, 
        train_script,
        '--data_path', args.data_path,
        '--output_dir', args.output_dir,
        '--window_size', str(args.window_size),
        '--embedding_dim', str(args.embedding_dim),
        '--max_sequence_length', str(args.max_sequence_length),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size)
    ]
    
    if args.use_class_weights:
        cmd.append('--use_class_weights')
    
    print(f"Running training script: {' '.join(cmd)}")
    subprocess.call(cmd)

def run_dashboard():
    """Run the real-time detection dashboard."""
    dashboard_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'real_time_detection.py')
    
    cmd = [
        "streamlit", 
        "run", 
        dashboard_script
    ]
    
    print(f"Running dashboard: {' '.join(cmd)}")
    subprocess.call(cmd)

def main():
    parser = argparse.ArgumentParser(description='Complaint Detection System')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train subcommand
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--data_path', type=str, required=True, help='Path to training data CSV file')
    train_parser.add_argument('--output_dir', type=str, default='output', help='Output directory for models and plots')
    train_parser.add_argument('--window_size', type=int, default=3, help='Number of utterances in each window')
    train_parser.add_argument('--embedding_dim', type=int, default=100, help='Dimension of embedding vectors')
    train_parser.add_argument('--max_sequence_length', type=int, default=128, help='Maximum sequence length')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    train_parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    train_parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for imbalanced data')
    
    # Dashboard subcommand
    dashboard_parser = subparsers.add_parser('dashboard', help='Run the real-time detection dashboard')
    
    # Install subcommand
    install_parser = subparsers.add_parser('install', help='Install required dependencies')
    
    args = parser.parse_args()
    
    if args.command == 'install':
        success = install_requirements()
        if success:
            print("Dependencies installed successfully!")
        else:
            print("Failed to install dependencies.")
    
    elif args.command == 'train':
        if not check_requirements():
            print("Missing dependencies. Run 'python run.py install' first.")
            return
        
        train_model(args)
    
    elif args.command == 'dashboard':
        if not check_requirements():
            print("Missing dependencies. Run 'python run.py install' first.")
            return
        
        run_dashboard()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 