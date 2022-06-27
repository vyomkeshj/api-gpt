import argparse
import requests
import numpy as np

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
    for p in np.linspace(0.70, 1.0, 10):
        for t in np.linspace(0.2, 0.9, 10):
            print(f"temp : {t} \n")
            print(f"top p : {p} \n")

            payload = {
                "question": question_on_insurance,
                "token_max_length": 340,
                "temperature": t,
                "top_p": p,
            }

            response = requests.post("http://localhost:5000/generate", params=payload).json()
            response = response['text']
            before, sep, after = response.partition('SELECT')
            query = sep + after
            query = find_between(query, "SELECT", "###")
            query = "SELECT" + query
            print("Query: " + query + '\n')
