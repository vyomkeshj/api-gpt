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
    payload = {
        "question": question_on_insurance,
        "token_max_length": 400,
        "temperature": 0.20,
        "top_p": 0.95,
    }

    response = requests.post("http://localhost:5000/generate", params=payload).json()
    response = response['text']
    print(response)
    before, sep, after = response.partition('SELECT')
    query = sep + after
    query = find_between(query, "SELECT", "###")
    query = "SELECT" + query
    print("Query: " + query + '\n')
