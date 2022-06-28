import streamlit as st
import time
import requests
import pandas as pd
import sqlite3


def main():
    DATA_CSV_FILE = './gistfile1.txt'
    data = pd.read_csv(DATA_CSV_FILE, sep=';')
    data.name = 'insurance_data'
    conn = sqlite3.connect("insurance.db")
    try:
        data.to_sql('insurance_data', conn)
    except:
        print("loaded old table!")

    query = conn.execute("SELECT * From insurance_data")
    cols = [column[0] for column in query.description]

    st.set_page_config(  # Alternate names: setup_page, page, layout
        layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
        initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
        page_title="Q. Research Edition",  # String or None. Strings get appended with "• Streamlit".
        page_icon=None,  # String, anything supported by st.image, or None.
    )

    question_col, data_col = st.columns((1, 1))
    st.title("Q. Research Edition")
    question_col.header("...")

    example = """How many people have claims for each auto model?"""
    question_on_insurance = st.text_area(
        "Ask your question!", example, max_chars=2000, height=150
    )
    temp = st.slider(
        "Increase the randomness if the default output is not right", 0.0, 0.18, 0.3)

    response = None
    with st.form(key="inputs"):
        submit_button = st.form_submit_button(label="Ask Q!")
        successful_run = False
        if submit_button:
            try_count = 5
            payload = {
                "question": question_on_insurance,
                "token_max_length": 350,
                "temperature": temp,
                "top_p": 1.0,
            }

            while try_count > 0
                query = requests.post("http://10.164.0.15:5000/run_query", params=payload)
                response = query.json()
                try:
                    st.markdown(response["prompt"])
                    result = pd.read_sql(f"SELECT {model_output}", conn)
                    st.dataframe(data=result, width=200, height=100)
                    try_count = 0
                    successful_run = True
                except:
                    try_count -= 1
            if not successful_run:
                st.markdown(response["Please try again with a slightly different question? :)"], unsafe_allow_html=True)
            else:
                st.text(f"Query done in {response['compute_time']:.3} s.")

    data_col.header("Data")
    insurance_table = pd.DataFrame.from_records(data=query.fetchall(), columns=cols)
    st.dataframe(data=insurance_table, width=200, height=100)

    if False:
        col1, col2, *rest = st.beta_columns([1, 1, 10, 10])

        def on_click_good():
            response["rate"] = "good"
            print(response)

        def on_click_bad():
            response["rate"] = "bad"
            print(response)

        col1.form_submit_button("👍", on_click=on_click_good)
        col2.form_submit_button("👎", on_click=on_click_bad)

    st.text("V0.0.2")


if __name__ == "__main__":
    main()
