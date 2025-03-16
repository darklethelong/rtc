import sys
import pandas as pd
from datetime import datetime, timedelta
import random
from utils.preprocessor import TextPreprocessor
from models.complaint_classifier import ComplaintClassifier

def generate_timestamps(num_utterances, start_time=None):
    if start_time is None:
        start_time = datetime.now().replace(microsecond=0)
    
    timestamps = []
    current_time = start_time
    
    for _ in range(num_utterances):
        timestamps.append(current_time.strftime("[%H:%M:%S]"))
        current_time += timedelta(seconds=random.randint(15, 45))
    
    return timestamps

def analyze_conversation_chunks(conversation, model, preprocessor, chunk_size=4):
    lines = conversation.split('\n')
    utterances = []
    
    # Extract valid utterances
    for line in lines:
        if line.strip() and (line.lower().startswith('agent:') or line.lower().startswith('caller:')):
            utterances.append(line.strip())
    
    # Generate timestamps
    timestamps = generate_timestamps(len(utterances))
    
    # Add timestamps to utterances
    timestamped_utterances = [
        f"{timestamp} {utterance}"
        for timestamp, utterance in zip(timestamps, utterances)
    ]
    
    # Process chunks
    chunks = []
    chunk_ratings = []
    
    for i in range(0, len(timestamped_utterances), chunk_size):
        chunk = timestamped_utterances[i:i+chunk_size]
        chunk_text = '\n'.join(chunk)
        chunks.append(chunk_text)
        
        # Process chunk for prediction
        processed_text, features = preprocessor.prepare_conversation(chunk_text)
        sequences = preprocessor.texts_to_sequences([processed_text])
        
        # Get prediction
        prediction = model.predict(sequences)[0][0]
        
        # Analyze complaint intensity
        if prediction > 0.7:
            rating = "High"
        elif prediction > 0.4:
            rating = "Moderate"
        elif prediction > 0.2:
            rating = "Mild"
        else:
            rating = "None"
            
        chunk_ratings.append(rating)
    
    # Calculate overall complaint percentage
    caller_utterances = [u for u in utterances if u.lower().startswith('caller:')]
    complaint_utterances = sum(1 for i, u in enumerate(caller_utterances) if chunk_ratings[i//chunk_size] != "None")
    complaint_percentage = (complaint_utterances / len(caller_utterances)) * 100
    
    # Generate markdown output
    output = ["# Call Center Complaint Conversation\n"]
    output.append("## Complete Conversation with Timestamps\n")
    
    for utterance in timestamped_utterances:
        output.append(f"**{utterance}**\n")
    
    output.append("\n## Conversation Split into Chunks (4 consecutive utterances per chunk) with Complaint Labels\n")
    
    for i, (chunk, rating) in enumerate(zip(chunks, chunk_ratings), 1):
        output.append(f"### Chunk {i}")
        for line in chunk.split('\n'):
            output.append(f"**{line}**")
            if line.lower().startswith('caller:'):
                output.append(f"*[Complaint: {'Yes' if rating != 'None' else 'No'}"
                            f"{f' - {rating} intensity' if rating != 'None' else ''}]*")
        output.append(f"\n**Chunk {i} Complaint Rating: {rating}**\n")
    
    output.append("## Analysis of Complaint Percentage\n")
    output.append(f"### Total Caller Utterances: {len(caller_utterances)}")
    output.append(f"- Utterances with complaints: {complaint_utterances}")
    for rating in ["High", "Moderate", "Mild"]:
        count = sum(1 for r in chunk_ratings if r == rating)
        output.append(f"  - {rating} intensity complaints: {count}")
    output.append(f"- Utterances without complaints: {len(caller_utterances) - complaint_utterances}\n")
    
    output.append("### Complaint Percentage Calculation:")
    output.append("(Number of caller utterances with complaints / Total number of caller utterances) × 100%")
    output.append(f"= ({complaint_utterances}/{len(caller_utterances)}) × 100% = {complaint_percentage:.2f}%\n")
    
    return '\n'.join(output)

def main():
    if len(sys.argv) != 3:
        print("Usage: python analyze_conversation.py <conversation_file> <model_path>")
        sys.exit(1)
        
    conversation_file = sys.argv[1]
    model_path = sys.argv[2]
    
    # Load model and initialize preprocessor
    model = ComplaintClassifier.load_model(model_path)
    preprocessor = TextPreprocessor()
    
    # Read conversation
    with open(conversation_file, 'r') as f:
        conversation = f.read()
    
    # Analyze conversation
    output = analyze_conversation_chunks(conversation, model, preprocessor)
    
    # Save output
    output_file = 'conversation_analysis.md'
    with open(output_file, 'w') as f:
        f.write(output)
    
    print(f"Analysis saved to {output_file}")

if __name__ == "__main__":
    main() 