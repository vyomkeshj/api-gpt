import streamlit as st
import time
import requests


def main():
    st.set_page_config(  # Alternate names: setup_page, page, layout
        layout="wide",  # Can be "centered" or "wide". In the future also "dashboard", etc.
        initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
        page_title="Q. Research Edition",  # String or None. Strings get appended with "• Streamlit".
        page_icon=None,  # String, anything supported by st.image, or None.
    )
    st.title("Query Insurance Dataset")
    """Ask: "show me the data" to get a peek at the dataset"""

    inp = st.text_area(
        "Ask your question!", example, max_chars=2000, height=150
    )

    temp = st.slider(
        "Increase the randomness if the default output is not right",
        0.0,
        1.5,
        1.0 if rec < 2 else 0.35,
        0.05,
    )

    response = None
    with st.form(key="inputs"):
        submit_button = st.form_submit_button(label="Generate!")

        if submit_button:
            payload = {
                "context": inp,
                "token_max_length": length,
                "temperature": temp,
                "top_p": 1.0,
            }

            query = requests.post("http://10.164.0.15:5000/run_query", params=payload)
            response = query.json()

            st.markdown(response["prompt"])
            st.markdown(response["query"])
            st.markdown(response["html"], unsafe_allow_html=True)

            st.text(f"Generation done in {response['compute_time']:.3} s.")

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
