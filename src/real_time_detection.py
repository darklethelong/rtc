import os
import sys
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_preprocessing import ConversationPreprocessor
from src.model import ComplaintDetector

class RealTimeComplaintDetector:
    def __init__(self, model_path, tokenizer_path, window_size=3):
        """
        Initialize the real-time complaint detector.
        
        Args:
            model_path (str): Path to the trained model
            tokenizer_path (str): Path to the tokenizer
            window_size (int): Number of utterances to consider as context window
        """
        self.window_size = window_size
        self.conversation_history = []
        self.complaint_scores = []
        self.timestamps = []
        
        # Load tokenizer
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        
        # Initialize preprocessor
        self.preprocessor = ConversationPreprocessor(window_size=window_size)
        self.preprocessor.tokenizer = self.tokenizer
        
        # Load model
        self.detector = ComplaintDetector()
        self.detector.load_model(model_path)
        
    def process_utterance(self, text, speaker):
        """
        Process a new utterance and update the complaint score.
        
        Args:
            text (str): The utterance text
            speaker (str): The speaker (either 'Agent' or 'Caller')
            
        Returns:
            float: The complaint probability score
        """
        # Clean text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Create utterance dict
        utterance = {
            'speaker': speaker,
            'text': cleaned_text
        }
        
        # Process utterance
        padded_sequence, feature_array, self.conversation_history = self.preprocessor.process_real_time_utterance(
            utterance, 
            self.conversation_history
        )
        
        # Make prediction
        prediction = self.detector.predict(padded_sequence, feature_array)[0][0]
        
        # Store prediction and timestamp
        self.complaint_scores.append(float(prediction))
        self.timestamps.append(datetime.now())
        
        return prediction
    
    def get_history(self):
        """Get the history of complaint scores and timestamps."""
        return {
            'scores': self.complaint_scores,
            'timestamps': self.timestamps
        }
    
    def reset(self):
        """Reset the conversation history and complaint scores."""
        self.conversation_history = []
        self.complaint_scores = []
        self.timestamps = []

def create_dashboard():
    """Create a Streamlit dashboard for real-time complaint detection."""
    st.set_page_config(
        page_title="Real-Time Complaint Detection",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("Real-Time Complaint Detection Dashboard")
    
    # Sidebar for configuration
    st.sidebar.title("Configuration")
    
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="../output/models/complaint_detector.h5"
    )
    
    tokenizer_path = st.sidebar.text_input(
        "Tokenizer Path", 
        value="../output/models/tokenizer.pickle"
    )
    
    window_size = st.sidebar.slider(
        "Window Size", 
        min_value=1, 
        max_value=10, 
        value=3
    )
    
    # Initialize detector if paths are valid
    detector = None
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        try:
            detector = RealTimeComplaintDetector(
                model_path=model_path,
                tokenizer_path=tokenizer_path,
                window_size=window_size
            )
            st.sidebar.success("Model loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
    else:
        st.sidebar.warning("Model or tokenizer path not found. Please provide valid paths.")
    
    # Initialize session state for conversation history
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    
    if 'complaint_probs' not in st.session_state:
        st.session_state.complaint_probs = []
    
    if 'timestamps' not in st.session_state:
        st.session_state.timestamps = []
    
    # Create input form
    st.subheader("Conversation Input")
    
    col1, col2 = st.columns(2)
    
    with col1:
        speaker = st.radio("Speaker", ["Agent", "Caller"])
    
    with col2:
        utterance = st.text_area("Utterance", height=100)
    
    # Create button row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Submit", key="submit"):
            if detector and utterance.strip():
                # Process utterance
                probability = detector.process_utterance(utterance, speaker)
                
                # Add to session state
                st.session_state.conversation.append({
                    'speaker': speaker,
                    'text': utterance,
                    'timestamp': datetime.now(),
                    'complaint_prob': probability
                })
                
                st.session_state.complaint_probs = detector.complaint_scores.copy()
                st.session_state.timestamps = [ts.strftime("%H:%M:%S") for ts in detector.timestamps]
    
    with col2:
        if st.button("Reset", key="reset"):
            if detector:
                detector.reset()
            st.session_state.conversation = []
            st.session_state.complaint_probs = []
            st.session_state.timestamps = []
    
    with col3:
        st.download_button(
            label="Export Conversation",
            data=pd.DataFrame(st.session_state.conversation).to_csv(index=False),
            file_name=f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            disabled=len(st.session_state.conversation) == 0
        )
    
    # Display conversation history
    st.subheader("Conversation History")
    
    if st.session_state.conversation:
        conversation_df = pd.DataFrame(st.session_state.conversation)
        st.dataframe(
            conversation_df[['speaker', 'text', 'complaint_prob']],
            use_container_width=True
        )
    else:
        st.info("No conversation history yet. Start by submitting an utterance.")
    
    # Display complaint probability chart
    st.subheader("Complaint Probability Over Time")
    
    if st.session_state.complaint_probs:
        fig = go.Figure()
        
        # Add complaint probability line
        fig.add_trace(go.Scatter(
            x=st.session_state.timestamps,
            y=st.session_state.complaint_probs,
            mode='lines+markers',
            name='Complaint Probability',
            line=dict(width=2, color='blue'),
            marker=dict(size=8)
        ))
        
        # Add threshold line
        fig.add_trace(go.Scatter(
            x=[st.session_state.timestamps[0], st.session_state.timestamps[-1]],
            y=[0.5, 0.5],
            mode='lines',
            name='Threshold (0.5)',
            line=dict(dash='dash', width=1, color='red')
        ))
        
        # Update layout
        fig.update_layout(
            title='Real-Time Complaint Detection',
            xaxis_title='Time',
            yaxis_title='Complaint Probability',
            yaxis=dict(range=[0, 1]),
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data to display yet. Start by submitting an utterance.")
    
    # Display latest complaint score
    if st.session_state.complaint_probs:
        latest_score = st.session_state.complaint_probs[-1]
        
        st.subheader("Current Complaint Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Complaint Probability",
                value=f"{latest_score:.2f}",
                delta=f"{latest_score - 0.5:.2f} from threshold"
            )
        
        with col2:
            if latest_score >= 0.5:
                st.error("⚠️ COMPLAINT DETECTED")
            else:
                st.success("✅ NO COMPLAINT DETECTED")

def main():
    create_dashboard()

if __name__ == "__main__":
    main() 