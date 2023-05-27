import json
import argparse
import qbmodel

# Import your QA model here

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def run_inference(data):
    # Perform inference using your QA model here
    # Replace the following print statement with your actual inference code
    for item in data:
        question = item["text"]
        answer = model.guess_and_buzz([question])
        ans, score = answer[0]
        print(ans, score)
        return ans

def main(args):
    data = load_data(args.data)
    run_inference(data)

if __name__ == '__main__':
    model = qbmodel.QuizBowlModel()
    parser = argparse.ArgumentParser(description='QA Inference')
    parser.add_argument('--data', type=str, help='Path to evaluation JSON file')
    args = parser.parse_args()
    main(args)
