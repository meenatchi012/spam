import joblib
import os
import sys
import numpy as np

# --- Configuration ---
MODEL_FILE = 'trained_spam_classifier_model.pkl'

def load_model():
    """Loads the trained machine learning pipeline."""
    try:
        # Check if the model file exists
        if not os.path.exists(MODEL_FILE):
            print(f"Error: Model file '{MODEL_FILE}' not found.")
            print("Please ensure it is in the same directory as this script.")
            sys.exit(1)

        # The pipeline must contain the text vectorizer AND the classifier
        pipeline = joblib.load(MODEL_FILE)
        print(f"Model '{MODEL_FILE}' loaded successfully.")
        return pipeline

    except Exception as e:
        print("-" * 50)
        print(f"CRITICAL ERROR: Failed to load model. Details: {e}")
        print("ACTION: This usually means a scikit-learn version mismatch. Check the library version you used for training.")
        print("-" * 50)
        sys.exit(1)

def predict_spam(pipeline, text_input):
    """Makes a prediction and prints the formatted result."""
    
    # The pipeline expects a list containing the single text string: ['Your text']
    data_to_predict = [text_input]
    # 

    try:
        # Use the full pipeline to transform the text (create features) and predict
        prediction_result = pipeline.predict(data_to_predict)[0]
        
        # --- Output Formatting ---
        # Assuming 1 is SPAM/Positive and 0 is HAM/Negative
        if prediction_result == 1 or str(prediction_result).lower() in ('spam', '1'):
            label = "SPAM"
            color_code = '\033[91m'  # Red
        else:
            label = "HAM (Not Spam)"
            color_code = '\033[92m'  # Green
        
        reset_code = '\033[0m'
        
        print("\n" + "=" * 50)
        print(f"CLASSIFICATION: {color_code}{label}{reset_code}")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"\n[Prediction Error] Could not classify message. Details: {e}")

def main():
    """Main function to run the CLI tool."""
    pipeline = load_model()

    print("\n--- Text Classification CLI Tool ---")
    print("Enter a message to check (or type 'quit' to exit).")
    
    while True:
        try:
            # Multi-line input for longer messages
            print("\nEnter SMS (Press Enter, then Ctrl+D or Ctrl+Z/Enter to finish):")
            # Read all lines until EOF (Ctrl+D on Linux/Mac, Ctrl+Z then Enter on Windows)
            user_input = sys.stdin.read().strip()
            
            if user_input.lower() in ('quit', 'exit'):
                print("Exiting tool. Goodbye!")
                break
            
            if user_input:
                predict_spam(pipeline, user_input)
                
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nExiting tool. Goodbye!")
            break
        except EOFError:
            # This is often triggered when using sys.stdin.read()
            print("\nExiting tool. Goodbye!")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break

if __name__ == "__main__":
    main()
