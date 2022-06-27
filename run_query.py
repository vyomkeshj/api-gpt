import argparse
import requests


def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, help="Your Query.")
    args = parser.parse_args()
    return args


def find_between(s, first, last):
    try:
        start = s.index(first) + len(first)
        end = s.index(last, start)
        return s[start:end]
    except ValueError:
        return ""


if __name__ == "__main__":
    args = parse_args()

    question_on_insurance = args.question
    for i in np.linspace(0.6, 1.8, 20):
        print(f"{i} \n")
        payload = {
            "question": question_on_insurance,
            "token_max_length": 340,
            "temperature": 0.90,
            "top_p": 0.95,
            "num_beams": 5
        }

    response = requests.post("http://localhost:5000/generate", params=payload).json()
    response = response['text']
    before, sep, after = response.partition('SELECT')
    query = sep + after
    query = find_between(query, "SELECT", "###")
    query = "SELECT" + query
    print("Query: " + query + '\n')
