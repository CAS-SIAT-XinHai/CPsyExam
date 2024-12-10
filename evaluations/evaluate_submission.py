import argparse
import json
import numpy as np

def is_single_choice(subject_name):
    """Determine if a subject is single choice based on its name"""
    return "单项选择题" in subject_name

def calculate_accuracy(predictions, answers):
    """Calculate accuracy for a specific shot setting"""
    metrics = {
        'MCQA': [],  # Single choice accuracy
        'MRQA': [],  # Multiple choice accuracy
    }
    
    # Create lookup dictionaries for faster matching
    prediction_lookup = {}
    answer_lookup = {}
    
    # Process predictions
    for subject, preds in predictions.items():
        prediction_lookup[subject] = {str(p['question_id']): p['prediction'] for p in preds}
    
    # Process answers
    for subject, ans in answers.items():
        answer_lookup[subject] = {str(a['question_id']): a['answer'] for a in ans}
    
    # Calculate metrics for each subject
    for subject in prediction_lookup:
        if subject not in answer_lookup:
            print(f"Warning: No answers found for subject {subject}")
            continue
        
        pred_dict = prediction_lookup[subject]
        ans_dict = answer_lookup[subject]
        
        # Match predictions with answers
        correct = 0
        total = 0
        
        for q_id in ans_dict:
            if q_id in pred_dict:
                total += 1
                if pred_dict[q_id] == ans_dict[q_id]:
                    correct += 1
        
        if total > 0:
            accuracy = (correct / total) * 100
            if is_single_choice(subject):
                metrics['MCQA'].append(accuracy)
            else:
                metrics['MRQA'].append(accuracy)
    
    # Calculate averages
    mcqa = np.mean(metrics['MCQA']) if metrics['MCQA'] else 0
    mrqa = np.mean(metrics['MRQA']) if metrics['MRQA'] else 0
    
    # Print detailed metrics
    print(f"\nDetailed metrics:")
    print(f"Single choice accuracies: {metrics['MCQA']}")
    print(f"Multiple choice accuracies: {metrics['MRQA']}")
    
    return {
        'MCQA': mcqa,
        'MRQA': mrqa,
        'Avg': (mcqa + mrqa) / 2
    }

def calculate_metrics(predictions, answers):
    """Calculate metrics for both zero-shot and few-shot settings"""
    # Get zero-shot and five-shot predictions from the new format
    zero_shot_preds = predictions.get('zero_shot', {})
    five_shot_preds = predictions.get('five_shot', {})
    
    # Calculate metrics for each setting
    print("\nZero-shot evaluation:")
    zero_shot_metrics = calculate_accuracy(zero_shot_preds, answers)
    
    print("\nFive-shot evaluation:")
    five_shot_metrics = calculate_accuracy(five_shot_preds, answers)
    
    # Format final metrics
    final_metrics = {
        'MCQA (Zero)': zero_shot_metrics['MCQA'],
        'MRQA (Zero)': zero_shot_metrics['MRQA'],
        'MCQA (Five)': five_shot_metrics['MCQA'],
        'MRQA (Five)': five_shot_metrics['MRQA'],
        'Avg (Zero)': zero_shot_metrics['Avg'],
        'Avg (Five)': five_shot_metrics['Avg']
    }
    
    return final_metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate CPsyExam submission')
    parser.add_argument('--submission_file', type=str, required=True,
                        help='Path to submission file (merged results)')
    parser.add_argument('--answer_file', type=str, required=True,
                        help='Path to answer file')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to save evaluation metrics')
    args = parser.parse_args()
    
    # Load files
    with open(args.submission_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    with open(args.answer_file, 'r', encoding='utf-8') as f:
        answers = json.load(f)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, answers)
    
    # Print metrics
    print("\nEvaluation Results:")
    print(f"MCQA (Zero): {metrics['MCQA (Zero)']:.2f}%")
    print(f"MRQA (Zero): {metrics['MRQA (Zero)']:.2f}%")
    print(f"MCQA (Five): {metrics['MCQA (Five)']:.2f}%")
    print(f"MRQA (Five): {metrics['MRQA (Five)']:.2f}%")
    print(f"Avg (Zero): {metrics['Avg (Zero)']:.2f}%")
    print(f"Avg (Five): {metrics['Avg (Five)']:.2f}%")
    
    # Save metrics
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nMetrics saved to: {args.output_file}")

if __name__ == '__main__':
    main() 