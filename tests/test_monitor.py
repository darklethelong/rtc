import unittest
import numpy as np
from datetime import datetime
import os
import json
import tempfile
from utils.real_time_monitor import ComplaintMonitor

class TestComplaintMonitor(unittest.TestCase):
    def setUp(self):
        """Set up test cases"""
        self.monitor = ComplaintMonitor(window_size=5)
        self.test_probabilities = [0.1, 0.5, 0.8, 0.3, 0.9]
        self.test_timestamps = [
            datetime.now().strftime("%H:%M:%S") 
            for _ in range(len(self.test_probabilities))
        ]
        
    def test_complaint_level_classification(self):
        """Test complaint level classification"""
        test_cases = [
            (0.1, 'None'),
            (0.3, 'Mild'),
            (0.5, 'Moderate'),
            (0.8, 'High'),
            (1.0, 'High')
        ]
        
        for prob, expected_level in test_cases:
            level = self.monitor.get_complaint_level(prob)
            self.assertEqual(level, expected_level)
            
    def test_window_size_limit(self):
        """Test sliding window functionality"""
        # Add more points than window size
        for prob, timestamp in zip(self.test_probabilities, self.test_timestamps):
            self.monitor.update(prob, timestamp)
            
        # Check if only window_size points are kept
        self.assertEqual(len(self.monitor.probabilities), min(5, len(self.test_probabilities)))
        
    def test_session_save_load(self):
        """Test session saving and loading"""
        # Add some data
        for prob, timestamp in zip(self.test_probabilities, self.test_timestamps):
            self.monitor.update(prob, timestamp)
            
        # Save session
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp:
            self.monitor.save_session(tmp.name)
            
            # Create new monitor and load session
            new_monitor = ComplaintMonitor(window_size=5)
            new_monitor.load_session(tmp.name)
            
            # Compare data
            self.assertEqual(len(new_monitor.probabilities), len(self.monitor.probabilities))
            self.assertEqual(len(new_monitor.timestamps), len(self.monitor.timestamps))
            self.assertEqual(len(new_monitor.complaint_levels), len(self.monitor.complaint_levels))
            
        # Clean up
        os.unlink(tmp.name)
        
    def test_threshold_ranges(self):
        """Test threshold ranges for complaint levels"""
        self.assertIn('High', self.monitor.threshold_ranges)
        self.assertIn('Moderate', self.monitor.threshold_ranges)
        self.assertIn('Mild', self.monitor.threshold_ranges)
        self.assertIn('None', self.monitor.threshold_ranges)
        
        # Test range values
        self.assertEqual(self.monitor.threshold_ranges['High'][0], 0.7)
        self.assertEqual(self.monitor.threshold_ranges['Moderate'][0], 0.4)
        self.assertEqual(self.monitor.threshold_ranges['Mild'][0], 0.2)
        self.assertEqual(self.monitor.threshold_ranges['None'][0], 0.0)
        
if __name__ == '__main__':
    unittest.main() 