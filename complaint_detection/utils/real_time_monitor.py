import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pandas as pd
import seaborn as sns
from IPython.display import clear_output
import json

class ComplaintMonitor:
    def __init__(self, window_size=10):
        """
        Initialize the real-time complaint monitor
        
        Args:
            window_size (int): Number of recent points to show in the sliding window
        """
        self.window_size = window_size
        self.timestamps = []
        self.probabilities = []
        self.labels = []
        self.complaint_levels = []
        self.threshold_ranges = {
            'High': (0.7, 1.0),
            'Moderate': (0.4, 0.7),
            'Mild': (0.2, 0.4),
            'None': (0.0, 0.2)
        }
        
        # Setup the plot style
        plt.style.use('seaborn')
        self.setup_plot()
        
    def setup_plot(self):
        """Initialize the plot layout"""
        self.fig = plt.figure(figsize=(15, 8))
        self.gs = self.fig.add_gridspec(2, 2)
        
        # Probability timeline
        self.ax1 = self.fig.add_subplot(self.gs[0, :])
        # Current status
        self.ax2 = self.fig.add_subplot(self.gs[1, 0])
        # Complaint distribution
        self.ax3 = self.fig.add_subplot(self.gs[1, 1])
        
        plt.tight_layout(pad=3.0)
        
    def get_complaint_level(self, probability):
        """Determine complaint level based on probability"""
        for level, (lower, upper) in self.threshold_ranges.items():
            if lower <= probability < upper:
                return level
        return 'High' if probability >= 1.0 else 'None'
        
    def update(self, probability, timestamp=None):
        """
        Update the monitor with new data
        
        Args:
            probability (float): Complaint probability from the model
            timestamp (str, optional): Timestamp of the utterance
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
        self.timestamps.append(timestamp)
        self.probabilities.append(probability)
        self.complaint_levels.append(self.get_complaint_level(probability))
        
        # Keep only the recent window
        if len(self.timestamps) > self.window_size:
            self.timestamps = self.timestamps[-self.window_size:]
            self.probabilities = self.probabilities[-self.window_size:]
            self.complaint_levels = self.complaint_levels[-self.window_size:]
            
        self.plot_update()
        
    def plot_update(self):
        """Update all plots"""
        # Clear all axes
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Plot 1: Probability Timeline
        self._plot_timeline()
        
        # Plot 2: Current Status
        self._plot_current_status()
        
        # Plot 3: Complaint Distribution
        self._plot_distribution()
        
        plt.tight_layout()
        clear_output(wait=True)
        plt.show()
        
    def _plot_timeline(self):
        """Plot the probability timeline"""
        # Create color map for complaint levels
        colors = ['red' if p >= 0.7 else 'orange' if p >= 0.4 
                 else 'yellow' if p >= 0.2 else 'green' for p in self.probabilities]
        
        self.ax1.plot(self.timestamps, self.probabilities, 'b-', alpha=0.3)
        self.ax1.scatter(self.timestamps, self.probabilities, c=colors, s=100)
        
        # Add threshold lines
        self.ax1.axhline(y=0.7, color='r', linestyle='--', alpha=0.3, label='High')
        self.ax1.axhline(y=0.4, color='orange', linestyle='--', alpha=0.3, label='Moderate')
        self.ax1.axhline(y=0.2, color='y', linestyle='--', alpha=0.3, label='Mild')
        
        self.ax1.set_ylim(-0.1, 1.1)
        self.ax1.set_title('Complaint Probability Timeline')
        self.ax1.set_xlabel('Time')
        self.ax1.set_ylabel('Probability')
        self.ax1.legend()
        
        # Rotate x-axis labels for better readability
        self.ax1.tick_params(axis='x', rotation=45)
        
    def _plot_current_status(self):
        """Plot the current complaint status"""
        if not self.probabilities:
            return
            
        current_prob = self.probabilities[-1]
        current_level = self.complaint_levels[-1]
        
        # Create a gauge-like visualization
        self.ax2.add_patch(plt.Circle((0.5, 0.5), 0.3, color='lightgray'))
        self.ax2.add_patch(plt.Circle((0.5, 0.5), 0.28, color='white'))
        
        # Color based on complaint level
        color = {'High': 'red', 'Moderate': 'orange', 
                'Mild': 'yellow', 'None': 'green'}[current_level]
                
        self.ax2.text(0.5, 0.65, f'Current Status', 
                     ha='center', va='center', fontsize=12)
        self.ax2.text(0.5, 0.5, f'{current_level}', 
                     ha='center', va='center', fontsize=15, color=color)
        self.ax2.text(0.5, 0.35, f'{current_prob:.2f}', 
                     ha='center', va='center', fontsize=12)
        
        self.ax2.set_xlim(0, 1)
        self.ax2.set_ylim(0, 1)
        self.ax2.axis('off')
        
    def _plot_distribution(self):
        """Plot the distribution of complaint levels"""
        if not self.complaint_levels:
            return
            
        level_counts = pd.Series(self.complaint_levels).value_counts()
        colors = ['red', 'orange', 'yellow', 'green']
        
        # Create bar plot
        bars = self.ax3.bar(level_counts.index, level_counts.values, 
                           color=[colors[list(self.threshold_ranges.keys()).index(level)] 
                                 for level in level_counts.index])
        
        self.ax3.set_title('Complaint Level Distribution')
        self.ax3.set_ylabel('Count')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            self.ax3.text(bar.get_x() + bar.get_width()/2., height,
                         f'{int(height)}',
                         ha='center', va='bottom')
                         
    def save_session(self, filepath):
        """Save the monitoring session data"""
        session_data = {
            'timestamps': self.timestamps,
            'probabilities': self.probabilities,
            'complaint_levels': self.complaint_levels
        }
        
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
            
    def load_session(self, filepath):
        """Load a previous monitoring session"""
        with open(filepath, 'r') as f:
            session_data = json.load(f)
            
        self.timestamps = session_data['timestamps']
        self.probabilities = session_data['probabilities']
        self.complaint_levels = session_data['complaint_levels']
        
        # Update the visualization
        self.plot_update() 