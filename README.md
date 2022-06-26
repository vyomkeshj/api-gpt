* Streamlit web app at 0.0.0.0:8000/ 
* The proper API, documented at 0.0.0.0:5000/docs

## Open API endpoints 🔓

These are the endpoints of the public API and require no authentication.
Click on each to see the parameters!

#### Model testing

* [generate](docs/generate.md) : `POST /generate/`

## Using the API 🔥

* Python:

```python
import requests

def find_between(s, first, last):
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""
    
question_on_insurance = "How many people own a Saab?"
    
payload = {
    "question": question_on_insurance,
    "token_max_length": 340,
    "temperature": 0.90,
    "top_p": 0.95,
}

response = requests.post("http://localhost:5000/generate", params=payload).json()
response  = response['text']
before, sep, after = response.partition('SELECT')
query = sep + after
query = find_between(query, "SELECT", "###")
query = "SELECT" + query
print("Query: " + query + '\n')
```

## Deployment of the API server

Just SSH into a TPU VM. This code was tested on both the v2-8 and v3-8 variants.

Activate the conda environment, then dowload weights:
```
conda activate ml_exp
gsutil cp -r gs://gpt-j-trainer-sql/sql_combined_slim_f16/step_201 ./
```

Change `serve.py` with the downloaded slim model path
```
network.state = read_ckpt(network.state, "./step_201/", devices.shape[1])
```

And just run
```
python3 serve.py
```

Then, you can go to http://localhost:5000/docs and use the API!

## Deploy the streamlit dashboard

```
python3 -m streamlit run streamlit_app.py --server.port 8000
```
