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
context = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."
payload = {
    "context": context,
    "token_max_length": 512,
    "temperature": 1.0,
    "top_p": 0.9,
}
response = requests.post("http://localhost:5000/generate", params=payload).json()
print(response)
```

* Python (zero-shot classification):

```python
import requests
payload = { 
    "sequence" : "The movie started slow, but in the end was absolutely amazing!", 
    "labels" : "positive,neutral,negative"}
response = requests.post("http://localhost:5000/classify", params=payload).json()
print(response)
```

* Bash:

```bash
curl -X 'POST' \
  'http://localhost:5000/generate?context=In%20a%20shocking%20finding%2C%20scientists%20discovered%20a%20herd%20of%20unicorns%20living%20in%20a%20remote%2C%20previously%20unexplored%20valley%2C%20in%20the%20Andes%20Mountains.%20Even%20more%20surprising%20to%20the%20researchers%20was%20the%20fact%20that%20the%20unicorns%20spoke%20perfect%20English.&token_max_length=512&temperature=1&top_p=0.9' \
  -H 'accept: application/json' \
  -d ''
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
