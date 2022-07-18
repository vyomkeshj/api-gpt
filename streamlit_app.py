import streamlit as st
import requests
import pandas as pd
import sqlite3
import nlpcloud

temperature = 0.18
top_p = 0.96

header = """###Postgre SQL tables, with their properties:"""
#default_schema = """# insurance_data: ("months_as_customer", "age", "policy_number", "policy_bind_date", "policy_state", "policy_csl", "policy_deductable", "policy_annual_premium", "umbrella_limit", "insured_zip", "insured_sex", "insured_education_level", "insured_occupation", "insured_hobbies", "insured_relationship", "capital_gains", "capital_loss", "incident_date", "incident_type", "collision_type", "incident_severity", "authorities_contacted", "incident_state", "incident_city", "incident_location", "incident_hour_of_the_day", "number_of_vehicles_involved", "property_damage", "bodily_injuries", "witnesses", "police_report_available", "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim", "auto_make", "auto_model", "auto_year", "fraud_reported"]"""
default_schema = """# insurance_data: (months_as_customer: INTEGER, age: INTEGER, policy_number: INTEGER, policy_bind_date: TEXT, policy_state: TEXT, policy_csl: TEXT, policy_deductable: INTEGER, policy_annual_premium: REAL, umbrella_limit: INTEGER, insured_zip: INTEGER, insured_sex: TEXT, insured_education_level: TEXT, insured_occupation: TEXT, insured_hobbies: TEXT, insured_relationship: TEXT, capital_gains: REAL, capital_loss: REAL, incident_date: TEXT, incident_type: TEXT, collision_type: TEXT, incident_severity: TEXT, authorities_contacted: TEXT, incident_state: TEXT, incident_city: TEXT, incident_location: TEXT, incident_hour_of_the_day"c, number_of_vehicles_involved: INTEGER, property_damage: TEXT, bodily_injuries:INTEGER, witnesses:INTEGER, police_report_available: TEXT, total_claim_amount: REAL, injury_claim: TEXT, property_claim: TEXT, vehicle_claim: TEXT, auto_make: TEXT, auto_model: TEXT, auto_year: INTEGER, fraud_reported: TEXT]"""

client = nlpcloud.Client("finetuned-gpt-neox-20b", "6b8a5bb4bc0bc846866168a32a86b0372683fe85", True)

kwargs = {'min_length': 0, 'max_length': 300, 'length_no_input': True,
          'remove_input': True, 'end_sequence': '###', 'top_p': 1,
          'temperature': 0.6, 'top_k': 50, 'repetition_penalty': 1,
          'length_penalty': 1, 'do_sample': True, 'early_stopping': True,
          'num_beams': 1, 'no_repeat_ngram_size': 0, 'num_return_sequences': 1,
          'bad_words': ["bad"], 'remove_end_sequence': True}


def get_generated(obj):
    return obj["generated_text"]


DATA_CSV_FILE = './gistfile1.txt'
data = pd.read_csv(DATA_CSV_FILE, sep=';')
data.name = 'insurance_data'
conn = sqlite3.connect("insurance.db")
try:
    data.to_sql('insurance_data', conn)
except:
    print("loaded old table!")

HIST_CSV_FILE = './history.csv'

example = """How many people love in policy state OH?"""


def main():
    history = pd.read_csv(HIST_CSV_FILE)
    query = conn.execute("SELECT * From insurance_data")
    cols = [column[0] for column in query.description]
    insurance_table = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)

    st.set_page_config(  # Alternate names: setup_page, page, layout
        layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
        initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
        page_title="Q. Research Edition",  # String or None. Strings get appended with "• Streamlit".
        page_icon=None,  # String, anything supported by st.image, or None.
    )
    hide_menu_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_menu_style, unsafe_allow_html=True)

    st.title("Q. Research Edition")
    st.text("Hint: please use real column names for improved results...")
    st.text("Hint: Try running again if the model fails the first time!")

    question_col, data_col = st.columns((1, 1))
    data_col.header("Insurance Data")
    data_col.dataframe(data=insurance_table, width=None, height=None)
    data_col.header("Query History")
    data_col.dataframe(data=history, width=None, height=None)

    schema = question_col.text_area(
        "Put the schema here!", default_schema, max_chars=1000, height=150
    )
    question_on_insurance = question_col.text_area(
        "Ask your question!", example, max_chars=2000, height=150
    )
    # allow_cheating = st.checkbox('Enable Cheating', value=True, help="""Allow the model to learn\
    #                                                                    from bigger models by cheating on this query.
    #                                                                    This allows us to train the model against\
    #                                                                     models that actually work on your query""")
    # st.text("The model tells you when it cheats!")

    temperature_val = question_col.slider("Increase the randomness", 0.18, 0.90, value=temperature)
    top_p_val = question_col.slider("Top p", 0.2, 1.0, value=top_p)
    raw_output = question_col.text(f"raw_output(our model):")

    response = None
    with question_col.form(key="inputs"):
        submit_button = st.form_submit_button(label="Ask Q!")
        successful_run = False
        if submit_button:
            try_count = 10
            payload = {
                "header": header,
                "schema": schema,
                "question": question_on_insurance,
                "token_max_length": 100,
                "stop_sequence": "#",
                "temperature": temperature_val,
                "top_p": top_p_val,
            }

            while try_count > 0:
                try:
                    query = requests.post("http://10.164.0.15:5000/run_query", params=payload)
                    response = query.json()
                    model_output = response["query"]
                    try:
                        raw_output.write(f"raw_output(our model): {model_output};")
                        result = pd.read_sql(f"{model_output};", conn)

                        # Save to history
                        my_dict = {'Query': question_on_insurance,
                                   'Response': f"""{model_output}""",
                                   'has_cheated': 'False'}
                        history = history.append(my_dict, ignore_index=True)
                        question_col.dataframe(data=result, width=None, height=None)

                        try_count = 0
                        successful_run = True
                    except Exception as e:
                        print(f"failed to execute {e}")
                        try_count -= 1
                        successful_run = False
                except:
                    question_col.markdown("The api seems to be down!")
            if not successful_run:
                question_col.markdown("Please try using real column names when possible :)", unsafe_allow_html=True)

        # if allow_cheating and not successful_run:
        #     context_initial = f"{header}\n{schema}"
        #     neo_input = f"{context_initial}\n###{question_on_insurance}\nSELECT"
        #     model_output = get_generated(client.generation(f"{neo_input}", **kwargs))
        #     model_output = f"SELECT{model_output}"
        #     result = pd.read_sql(model_output, conn)
        #     question_col.dataframe(data=result, width=None, height=None)
        #     question_col.text(f"raw_output(cheated): {model_output}")
        #     my_dict = {'Query': question_on_insurance,
        #                'Response': f"""{model_output}""",
        #                'has_cheated': 'True'}
        #     history.append(my_dict, ignore_index=True)
        # else:

        else:
            history.to_csv(HIST_CSV_FILE, index=False)

    st.text("V0.0.2")


if __name__ == "__main__":
    main()
