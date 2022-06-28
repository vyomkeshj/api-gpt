import time

import jax
from jax.experimental import maps
import numpy as np
import optax
import transformers
import pandas as pd
import sqlite3

from mesh_transformer.checkpoint import read_ckpt
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer

from fastapi import FastAPI
import uvicorn

from typing import Optional

np.random.seed(0)

app = FastAPI()

DATA_CSV_FILE = './gistfile1.txt'
data = pd.read_csv(DATA_CSV_FILE, sep=';')
data.name = 'insurance_data'
conn = sqlite3.connect(
    "insurance.db")  # if the db does not exist, this creates a Any_Database_Name.db file in the current directory
# store your table in the database:
data.to_sql('insurance_data', conn)

params = {
    "layers": 28,
    "d_model": 4096,
    "n_heads": 16,
    "n_vocab": 50400,
    "norm": "layernorm",
    "pe": "rotary",
    "pe_rotary_dims": 64,
    "seq": 2048,
    "cores_per_replica": 8,
    "per_replica_batch": 1,
}

per_replica_batch = params["per_replica_batch"]
cores_per_replica = params["cores_per_replica"]
seq = params["seq"]

params["sampler"] = nucleaus_sample

# here we "remove" the optimizer parameters from the model (as we don't need them for inference)
params["optimizer"] = optax.scale(0)

mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
devices = np.array(jax.devices()).reshape(mesh_shape)

maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(devices, ("dp", "mp")), ())

tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")

total_batch = per_replica_batch * jax.device_count() // cores_per_replica

network = CausalTransformer(params)
network.state = read_ckpt(network.state, "./step_383500/", devices.shape[1])
del network.state["opt_state"]
network.state = network.move_xmap(network.state, np.zeros(cores_per_replica))

header = """###Postgres SQL tables, with their properties:"""
schema = """#Insurance_Data(months_as_customer,age,policy_number,policy_bind_date,policy_state,policy_csl,policy_deductable,policy_annual_premium,umbrella_limit,insured_zip,insured_sex,insured_education_level,insured_occupation,insured_hobbies,insured_relationship,capital_gains,capital_loss,incident_date,incident_type,collision_type,incident_severity,authorities_contacted,incident_state,incident_city,incident_location,incident_hour_of_the_day,number_of_vehicles_involved,property_damage,bodily_injuries,witnesses,police_report_available,total_claim_amount,injury_claim,property_claim,vehicle_claim,auto_make,auto_model,auto_year,fraud_reported)"""
context_initial = f"{header}\n{schema}"


@app.post("/generate")
async def generate(
        question: str,
        token_max_length: Optional[int] = 330,
        temperature: Optional[float] = 0.20,
        top_p: Optional[float] = 0.95,
        stop_sequence: Optional[str] = "\n",
):
    start = time.time()
    input = f"{context_initial}\n###{question}\nSELECT"
    print(input)

    tokens = tokenizer.encode(input)
    provided_ctx = len(tokens)
    if token_max_length + provided_ctx > 2048:
        return {"text": "Don't abuse the API, please."}
    pad_amount = seq - provided_ctx

    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
    batched_tokens = np.array([padded_tokens] * total_batch)
    length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

    output = network.generate(
        batched_tokens,
        length,
        token_max_length,
        {
            "top_p": np.ones(total_batch) * top_p,
            "temp": np.ones(total_batch) * temperature,
        },
    )

    text = tokenizer.decode(output[1][0][0, :, 0])

    # A simple technique to stop at stop_sequence without modifying the underlying model
    if stop_sequence is not None and stop_sequence in text:
        text = text.split(stop_sequence)[0] + stop_sequence

    response = {}
    response["model"] = "gpt-sql"
    response["compute_time"] = time.time() - start
    response["text"] = "SELECT" + text
    response["prompt"] = question
    response["token_max_length"] = token_max_length
    response["temperature"] = temperature
    response["top_p"] = top_p
    response["stop_sequence"] = stop_sequence

    return response


@app.post("/run_query")
async def generate(
        question: str,
        token_max_length: Optional[int] = 330,
        temperature: Optional[float] = 0.20,
        top_p: Optional[float] = 0.95,
        stop_sequence: Optional[str] = "\n",
        try_count: Optional[int] = 5,

):
    start = time.time()
    input = f"{context_initial}\n###{question}\nSELECT"

    tokens = tokenizer.encode(input)
    provided_ctx = len(tokens)
    if token_max_length + provided_ctx > 2048:
        return {"model_output": "Don't abuse the API, please."}
    pad_amount = seq - provided_ctx

    padded_tokens = np.pad(tokens, ((pad_amount, 0),)).astype(np.uint32)
    batched_tokens = np.array([padded_tokens] * total_batch)
    length = np.ones(total_batch, dtype=np.uint32) * len(tokens)

    response = {}

    while try_count > 0:
        output = network.generate(
            batched_tokens,
            length,
            token_max_length,
            {
                "top_p": np.ones(total_batch) * top_p,
                "temp": np.ones(total_batch) * temperature,
            },
        )

        model_output = tokenizer.decode(output[1][0][0, :, 0])

        # A simple technique to stop at stop_sequence without modifying the underlying model
        if stop_sequence is not None and stop_sequence in model_output:
            model_output = model_output.split(stop_sequence)[0]

        try:
            print(f"SELECT {model_output}")
            response["query"] = f"SELECT {model_output}

            result = pd.read_sql(f"SELECT {model_output}", conn)

            try_count = 0
            response["html"] = result.to_html()
        except:
            try_count -= 1
            print("Failed to execute query")

    response["model"] = "gpt-sql"
    response["compute_time"] = time.time() - start
    response["prompt"] = question
    response["token_max_length"] = token_max_length
    response["temperature"] = temperature
    response["top_p"] = top_p
    response["stop_sequence"] = stop_sequence

    return response


print("sql-model-serving")
uvicorn.run(app, host="0.0.0.0", port=5000)
