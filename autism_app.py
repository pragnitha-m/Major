import streamlit as st
import numpy as np
import pandas as pd
pip install joblib
import joblib
import pandas as pd

# Load the dataset
df = pd.read_csv("cleaned_train.csv")  # replace with the actual path to your dataset
country_columns = [col for col in df.columns if col.startswith('country_of_residence_')]
country_list = sorted([col.replace('country_of_residence_', '') for col in country_columns])
ethnicity_columns = [col for col in df.columns if col.startswith('ethnicity_')]
ethnicity_list = sorted([
    col.replace('ethnicity_', '')
    for col in ethnicity_columns
    if '?' not in col and 'others' not in col
])


st.set_page_config(page_title="ASD Detection", layout="wide")
st.markdown("""
<style>
    .stButton>button {
        background-color: #f0f2f6;
        border: none;
        border-radius: 8px;
        padding: 0.5em 1em;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #f7f7f7;
            color: black;  /* Set default text color to black */
            padding-left: 5vw;
            padding-right: 5vw;
        }
        header[data-testid="stHeader"] {
            background-color: black;
        }
        header[data-testid="stHeader"]::before {
            content: "🧠 Online Assessment for Autism Detection";
            display: block;
            color: white;  /* Only this text is white */
            font-size: 28px;
            font-weight: bold;
            padding: 20px;
            text-align: center;
        }
        .pink-box {
            background-color: #ffc0cb;
            padding: 50px 40px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 40px;
        }
        .pink-box h1 {
            font-size: 36px;
            margin-bottom: 25px;
        }
        .yellow-box {
            background-color: #fff3cd;
            padding: 40px;
            border-left: 8px solid #ffec99;
            border-radius: 10px;
            margin-bottom: 40px;
            font-size: 18px;
            line-height: 1.7;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        .stApp {
            background-color: #f7f7f7;
            color: black;
            padding-left: 5vw;
            padding-right: 5vw;
        }
        header[data-testid="stHeader"] {
            background-color: black;
        }
        header[data-testid="stHeader"]::before {
            content: "🧠 Online Assessment for Autism Detection";
            display: block;
            color: white;
            font-size: 28px;
            font-weight: bold;
            padding: 20px;
            text-align: center;
        }
        .pink-box {
            background-color: #ffc0cb;
            padding: 50px 40px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 40px;
        }
        .pink-box h1 {
            font-size: 36px;
            margin-bottom: 25px;
        }
        .yellow-box {
            background-color: #fff3cd;
            padding: 40px;
            border-left: 8px solid #ffec99;
            border-radius: 10px;
            margin-bottom: 40px;
            font-size: 18px;
            line-height: 1.7;
        }
        div[data-testid="column"] > div {
            background-color: #d9d9d9 !important; /* grey background for Q&A blocks */
            color: black !important; /* black text inside the question blocks */
            border-radius: 10px;
            padding: 10px;
        }
        .stAlert {
            background-color: #d9d9d9 !important;
            color: black !important; /* text in “You selected” success box */
        }
        .stButton>button {
            color: black !important;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        .stButton>button {
            background-color: #d9d9d9 !important;  /* grey background */
            color: black !important;              /* black text */
            border: none;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
    </style>
""", unsafe_allow_html=True)


# Take Your Quick Assessment section
st.markdown("""<div class="pink-box"><h1>Take Your Quick Assessment</h1></div>""", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])  # Adjust ratios as needed

with col1:
   st.image("wired-lineal-656-person-sign-protest-hover-pinch.gif") 

with col2:
    st.markdown("""
    <h4>Here’s how it works:</h4>
    <p>You indicate the behaviors that are making you concerned about your <b>ward</b> by answering a series of questions. The Symptom Checker analyzes your answers to give you a list of psychiatric or learning disorders that are associated with those symptoms.</p>
    <p>Since individual symptoms can reflect more than one disorder, this tool will give you a range of possibilities and guide you toward next steps. This tool cannot diagnose your <b>ward</b>.</p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background-color: #fff3cd; padding: 25px; border-left: 8px solid #ffec99; border-radius: 10px; font-size: 17px; line-height: 1.6; margin-top: 15px;">
        <p>Remember, this tool is not a substitute for a diagnostic evaluation by a medical or mental health professional. If you believe your <b>ward</b> has a psychiatric or learning disorder, please consult a professional.</p>
    </div>
    """, unsafe_allow_html=True)



# --- Model and Input Section ---
model = joblib.load("stacking_model.pkl")
scaler = joblib.load("scaler.pkl")

if 'answers' not in st.session_state:
    st.session_state.answers = [None] * 10

questions_by_group = {
    "Kids": ["Does your child ignore when you call their name?",
             "Is it difficult for you to get eye contact with your child?",
             "Does your child avoid pointing to indicate what they want?",
             "Does your child avoid pointing to share interest with you?",
             "Does your child avoid pretend or make-believe play?",
             "Does your child not follow where you're looking?",
             "If someone is upset, does your child not try to comfort them?",
             "Were your child’s first words delayed or unusual?",
             "Does your child avoid nodding for ‘yes’ or shaking head for ‘no’?",
             "Does your child stare at nothing without purpose?"],
    "Teens": ["Do you find it hard to make new friends?",
              "Do you prefer to be alone rather than with others?",
              "Do you struggle with understanding jokes or sarcasm?",
              "Do loud sounds or bright lights bother you?",
              "Do you stick to a routine and get upset if it's disrupted?",
              "Do you get deeply interested in specific topics?",
              "Do you find it hard to maintain eye contact?",
              "Do you often misunderstand social cues?",
              "Do you dislike changes in your environment?",
              "Do people say you speak in a flat or unusual tone?"],
    "Adults": ["Do you find social situations confusing?",
               "Do you prefer doing things the same way every time?",
               "Are you very sensitive to sounds or textures?",
               "Do you struggle with small talk or casual conversations?",
               "Do you focus intensely on a single interest?",
               "Do you avoid eye contact in conversations?",
               "Do you find it hard to understand other people's feelings?",
               "Do you notice details that others miss?",
               "Do people describe you as honest or blunt?",
               "Do you get overwhelmed in busy environments?"]
}

age_group = st.selectbox("Select Age Group", ["Select an option", "Kids", "Teens", "Adults"])
questions = []
if age_group != "Select an option":
    questions = questions_by_group[age_group]
    st.subheader("Behavioral Questions")

    for idx, question in enumerate(questions):
        st.markdown(f"**Q{idx+1}. {question}**")
        col1, col2 = st.columns(2)

        # Set current selection
        selected = st.session_state.answers[idx]

        with col1:
            if st.button("✅ Agree", key=f"agree_{idx}"):
                st.session_state.answers[idx] = 1
        with col2:
            if st.button("❌ Disagree", key=f"disagree_{idx}"):
                st.session_state.answers[idx] = 0

        # Visual feedback
        if st.session_state.answers[idx] == 1:
            st.markdown(f'<p style="color: white; background-color: green; padding: 0.3em 1em; border-radius: 6px; display: inline-block;">You selected: Agree ✅</p>', unsafe_allow_html=True)
        elif st.session_state.answers[idx] == 0:
            st.markdown(f'<p style="color: white; background-color: red; padding: 0.3em 1em; border-radius: 6px; display: inline-block;">You selected: Disagree ❌</p>', unsafe_allow_html=True)



            
if all(answer is not None for answer in st.session_state.answers):
    st.subheader("User Details")
    gender = st.selectbox("Gender", ["Select an option", "Female", "Male"])
    jaundice = st.selectbox("Jaundice at birth", ["Select an option", "No", "Yes"])
    test_taker = st.selectbox("Test taken by", ["Select an option", "Parent", "Self", "Caregiver", "Healthcare professional"])

    if age_group == "Kids":
        age = st.slider("Select Age", min_value=1, max_value=12, step=1)
        st.markdown("""
        <div class="slider-labels">
        <span></span><span>2</span><span>3</span><span>4</span><span>5</span><span>6</span><span>7</span><span>8</span><span>9</span><span>10</span><span>11</span><span></span>
        </div>
        <style>
        .slider-labels {
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            margin-top: -20px;
            padding: -1 13px;
        }
        </style>
        """, unsafe_allow_html=True)

    elif age_group == "Teens":
        age = st.slider("Select Age", min_value=13, max_value=19, step=1)
        st.markdown("""
        <div class="slider-labels">
        <span></span><span>14</span><span>15</span><span>16</span><span>17</span><span>18</span><span></span>
        </div>
        <style>
        .slider-labels {
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            margin-top: -20px;
            padding: -1 13px;
        }
        </style>
        """, unsafe_allow_html=True)

    else:  # Adults
        age = st.slider("Select Age", min_value=20, max_value=80, step=1)
        st.markdown("""
        <div class="slider-labels">
        <span></span><span>25</span><span>30</span><span>35</span><span>40</span><span>45</span><span>50</span><span>55</span><span>60</span><span>65</span><span>70</span><span>75</span><span></span>
        </div>
        <style>
        .slider-labels {
            display: flex;
            justify-content: space-between;
            font-size: 14px;
            margin-top: -20px;
            padding: -1 13px;
        }
        </style>
        """, unsafe_allow_html=True)





    ethnicity = st.selectbox("Ethnicity", ["Select an option"] + ethnicity_list)
    country = st.selectbox("Country of Residence", ["Select an option"] + country_list)


    if st.button("Submit"):
        if gender == "Select an option" or jaundice == "Select an option" or test_taker == "Select an option":
            st.warning("Please complete all fields before submitting.")
        else:
            input_data = {}

            # Behavioral scores
            for i in range(10):
                input_data[f"A{i+1}_Score"] = st.session_state.answers[i]

            input_data["age"] = age

            # Jaundice
            input_data[f"jaundice_at_birth_{jaundice.lower()}"] = 1
            for key in ['jaundice_at_birth_yes','jaundice_at_birth_no']:
                if key not in input_data:
                    input_data[key] = 0

            # Gender
            input_data[f"gender_{gender}"] = 1
            for key in ['gender_f', 'gender_m']:
                if key not in input_data:
                    input_data[key] = 0

            # Test taker
            input_data[f"test_taker_{test_taker}"] = 1
            for key in ['test_taker_Health care professional', 'test_taker_Others',
                        'test_taker_Parent', 'test_taker_Relative', 'test_taker_Self']:
                if key not in input_data:
                    input_data[key] = 0

            # Final column order
            columns = [f"A{i+1}_Score" for i in range(10)] + ['age',
                    'gender_f', 'gender_m',
                    'jaundice_at_birth_no', 'jaundice_at_birth_yes',
                    'test_taker_Health care professional', 'test_taker_Others',
                    'test_taker_Parent', 'test_taker_Relative', 'test_taker_Self']

            X_input = pd.DataFrame([input_data])[columns]
            X_scaled = scaler.transform(X_input)
            pred_proba = model.predict_proba(X_scaled)[0][1]
            percent = round(pred_proba * 100, 2)


            st.markdown("### Result")

            if percent > 50:
                st.markdown(
                    f"""
                    <div style='background-color:#FF4B4B; padding:15px; border-radius:10px; text-align:center;'>
                        <span style='color:white; font-size:20px;'>⚠️ High likelihood of Autism: {percent:.1f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif percent > 20:
                st.markdown(
                    f"""
                    <div style='background-color:#FACC15; padding:15px; border-radius:10px; text-align:center;'>
                        <span style='color:black; font-size:20px;'>🔎 Moderate signs of Autism: {percent:.1f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div style='background-color:#22C55E; padding:15px; border-radius:10px; text-align:center;'>
                        <span style='color:white; font-size:20px;'>✅ Low likelihood of Autism: {percent:.1f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

else:
    if age_group != "Select an option":
        st.markdown(
    """
    <style>
    .custom-info-box {
        background-color: #1E90FF;  /* or match your blue background */
        padding: 10px;
        border-radius: 8px;
        color: white;
        font-size: 16px;
        font-weight: 500;
    }
    </style>

    <div class="custom-info-box">
        Answer all 10 questions to proceed to user details.
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("""
    <style>
        .stButton>button:hover {
            border: 2px solid #333 !important;
        }
        .stButton>button:focus {
            background-color: #28a745 !important;
            color: white !important;
            border: 2px solid #28a745 !important;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        .stButton>button {
            background-color: #d9d9d9 !important;
            color: black !important;
            border: none;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
        .stButton>button:hover {
            border: 2px solid #333 !important;
        }
        .agree-button:focus {
            background-color: #28a745 !important;
            color: white !important;
            border: 2px solid #28a745 !important;
        }
        .disagree-button:focus {
            background-color: #dc3545 !important;
            color: white !important;
            border: 2px solid #dc3545 !important;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        .custom-agree-selected button {
            background-color: #28a745 !important; /* green */
            color: white !important;
        }
        .custom-disagree-selected button {
            background-color: #dc3545 !important; /* red */
            color: white !important;
        }
        .stButton>button {
            background-color: #d9d9d9 !important;
            color: black !important;
            border: none;
            border-radius: 8px;
            padding: 0.5em 1em;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
    label, .stSelectbox label {
        color: black !important;
    }
    div[data-baseweb="select"] {
        color: black !important;
    }
</style>
""", unsafe_allow_html=True)