import streamlit as st
import requests
import pandas as pd
import sqlite3

# import nlpcloud

temperature = 0.11
top_p = 0.96

header = """###Postgre SQL tables, with their properties are given below. The year is represented in the 
incident_date column """

default_schema = """# Insurance_Data("months_as_customer","policy_state": {OH, NY, ...}, "age", "collision_type": {
Side Collision, Rear Collision, Front Collision, Not Known}, "policy_number" : INTEGER, "incident_type" : {
Multi-vehicle Collision, Parked Car, Single Vehicle Collision, Vehicle Theft}, "policy_bind_date", "policy_csl", 
"policy_deductable", "policy_annual_premium", "umbrella_limit", "insured_zip", "insured_sex" : {MALE, FEMALE}, 
"insured_education_level" : {MD, PhD, College, JD, Masters, Associate}, "insured_occupation" : {craft-repair, sales, 
tech-support, ...}, "insured_hobbies": {sleeping, reading, board-games, ...}, "insured_relationship": {husband, 
other-relative, own-child, unmarried, wife, not-in-family, ...}, "capital_gains": REAL, "capital_loss": REAL, 
"incident_date": {Example: 2020-01-25 00:00:00.000000, Hint: YYYY-MM-DD HH:MM:SS}, "incident_severity": {Major 
Damage, Total Loss, Minor Damage, Trivial Damage}, "authorities_contacted": {Police, Fire, Ambulance, Other, None, 
...}, "incident_state": {WV, OH, NY, ...}, "incident_city":  {Columbus, Riverwood, Arlington}, "incident_location", 
"incident_hour_of_the_day": INTEGER, "number_of_vehicles_involved": INTEGER, "property_damage", "bodily_injuries": 
INTEGER, "witnesses": INTEGER, "police_report_available": {YES, NO, Not Known}, "total_claim_amount": REAL, 
"injury_claim": REAL, "property_claim": REAL, "vehicle_claim": REAL, "auto_make": {Saab, Mercedes, Accura, ...}, 
"auto_model":{E400, RAM, Tahoe, ...}, "auto_year": INTEGER, "fraud_reported": {Y, N}) """

kwargs = {'min_length': 0, 'max_length': 300, 'length_no_input': True,
          'remove_input': True, 'end_sequence': '###', 'top_p': 1,
          'temperature': 0.6, 'top_k': 50, 'repetition_penalty': 1,
          'length_penalty': 1, 'do_sample': True, 'early_stopping': True,
          'num_beams': 1, 'no_repeat_ngram_size': 0, 'num_return_sequences': 1,
          'bad_words': ["bad"], 'remove_end_sequence': True}

DATA_CSV_FILE = './gistfile1.txt'


def main():
    example = """What are the top 3 incident types by number of incidents?"""

    data = pd.read_csv(DATA_CSV_FILE, sep=';')
    data.name = 'insurance_data'
    conn = sqlite3.connect("insurance.db")
    try:
        data.to_sql('insurance_data', conn)
    except:
        print("loaded old table!")

    # history = pd.read_csv(HIST_CSV_FILE)
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

    question_col, answer_col = st.columns((1, 1))
    # answer_col.header("Insurance Data")
    question_on_insurance = question_col.text_area(
        "Ask your question!", example, max_chars=2000, height=150
    )
    # change the default question to the last question
    example = question_on_insurance

    temperature_val = question_col.slider("Increase the randomness", 0.18, 0.90, value=temperature)
    top_p_val = question_col.slider("Top p", 0.2, 1.0, value=top_p)
    raw_output = answer_col.text(f"raw_output(our model):")
    question_col.dataframe(data=insurance_table, width=None, height=None)

    response = None
    with question_col.form(key="inputs"):
        submit_button = st.form_submit_button(label="Ask Q!")
        successful_run = False
        if submit_button:
            try_count = 10
            payload = {
                "header": header,
                "schema": default_schema,
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
                        raw_output.write(f"Attempt #{try_count}")
                        raw_output.write(f"Generated Query: {model_output}")

                        result = pd.read_sql(f"{model_output}", conn)
                        raw_output.write(f"Generated Query: {model_output}")
                        answer_col.write(
                            f"Please try again with the same query and a higher temperature if ouput is incorrect!")

                        answer_col.dataframe(data=result, width=None, height=None)

                        try_count = 0
                        successful_run = True
                    except Exception as e:
                        print(f"failed to execute {e}")
                        try_count -= 1
                        successful_run = False
                except:
                    question_col.markdown("The api seems to be down!")
                    break
            if not successful_run:
                question_col.markdown("Please try using real column names when possible :)", unsafe_allow_html=True)

    st.text("V0.0.8")


if __name__ == "__main__":
    main()
