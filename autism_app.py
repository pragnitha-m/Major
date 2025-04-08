import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pandas as pd

# Load the dataset
df = pd.read_csv("C:/Users/varsh/Downloads/PROJECTS/MAJOR/cleaned_train.csv")  # replace with the actual path to your dataset
ethnicity_list = [
    "Asian", "Black", "Hispanic", "Latino", "Middle Eastern", "Pasifika", "South Asian",
    "Turkish", "White-European"
]

country_list = [
    "Afghanistan", "Australia", "Austria", "Bahamas", "Brazil", "Canada", "France", "India",
    "Iran", "Ireland", "Italy", "Jordan", "Kazakhstan", "Malaysia", "Netherlands", "New Zealand",
    "Russia", "South Africa", "Spain", "Sri Lanka", "United Arab Emirates", "United Kingdom",
    "United States", "Viet Nam"
]


st.set_page_config(page_title="ASD Detection", layout="wide")

st.markdown("""
<style>
/* Style for all buttons */
button[kind="primary"] {
    font-family: Cambria, serif !important;
    color: #393346 !important;
}

/* Override hover/focus colors if needed */
button[kind="primary"]:hover {
    color: #393346 !important;
}
</style>
""", unsafe_allow_html=True)

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

st.markdown("""
            <link href="https://fonts.googleapis.com/css2?family=Playfair+Display&display=swap" rel="stylesheet">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Bodoni+Moda&display=swap');
        

        .stApp {
            background-color: #ebe6ee;
            color: black;
            padding-left: 5vw;
            padding-right: 5vw;
        }
        header[data-testid="stHeader"] {
            background-color: #2ca48d;
        }
        header[data-testid="stHeader"]::before {
            content: "🧠 Online Assessment for Autism Detection";
            display: block;
            color: #393346;
            font-size: 28px;
            font-weight: bold;
            padding: 20px;
            text-align: center;
            font-family: 'Bodoni Moda', serif;
        }
        .pink-box {
            background: #f8c8c0;
            background-image: 
                radial-gradient(circle at 10% 20%, #fbe3df 15%, transparent 20%),
                radial-gradient(circle at 25% 40%, #f7d6cf 15%, transparent 20%),
                radial-gradient(circle at 40% 10%, #f5d0c7 20%, transparent 25%),
                radial-gradient(circle at 60% 60%, #fad2cb 15%, transparent 20%),
                radial-gradient(circle at 75% 30%, #f6d9d1 15%, transparent 20%),
                radial-gradient(circle at 85% 80%, #fbd7d2 15%, transparent 20%),
                radial-gradient(circle at 50% 50%, #fddad3 15%, transparent 20%),
                radial-gradient(circle at 90% 15%, #f9cdc3 20%, transparent 25%);
            background-repeat: no-repeat;
            background-size: cover;
            padding: 50px 40px;
            border-radius: 16px;
            text-align: center;
            margin-bottom: 40px;
            box-shadow: 0 10px 25px rgba(248, 200, 192, 0.5);
            border: 2px solid #f3b5aa;
            position: relative;
            overflow: hidden;
            transition: transform 0.3s ease-in-out;
        }

        .pink-box:hover {
            transform: scale(1.02);
        }

        .pink-box h1 {
            font-size: 36px;
            margin-bottom: 25px;
            color: #b16836;
            font-family: 'Playfair Display', serif;
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
st.markdown("""<div class="pink-box"><h1>Take Your Quick Assessment Now</h1></div>""", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])  # Adjust ratios as needed

with col1:
   st.image("C:/Users/varsh/Downloads/PROJECTS/MAJOR/wired-lineal-656-person-sign-protest-hover-pinch.gif") 

with col2:
    st.markdown("""
    <div style="font-family: Cambria, serif;">
        <h4 style="color: #393346;">Here’s how it works:</h4>
        <p>You can indicate the behaviors that concern you about your <b> ward or yourself </b>by answering a short series of behavioral questions. This Autism Detection Tool will analyze your responses to estimate the likelihood of Autism Spectrum Disorder (ASD) based on patterns observed in clinical datasets.</p>
        <p>Since many behaviors can be associated with different conditions, this tool provides a probability score rather than a diagnosis and offers guidance for next steps. It is not a substitute for a medical or psychological evaluation.</p>
    </div>
    <div style="background-color: #fff3cd; padding: 25px 30px; border-left: 6px solid #ffe39f; border-radius: 10px; font-size: 17px; line-height: 1.6; margin-top: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
        <p style="margin: 0; color: #393346; font-family: Cambria, serif;">
        Remember ,if you suspect that your <b> ward or yourself </b>may be showing signs of ASD or any other developmental condition, please consult a licensed medical or mental health professional for a complete assessment.
        </p>
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
st.markdown("""
<style>
    .select-label {
        font-family: 'Playfair Display', serif;
        font-size: 24px;
        font-weight: bold;
        color: #393346;
        margin-bottom: 0px;
    }
</style>
<div class="select-label">Select Age Group</div>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
        /* Dropdown container */
        div[data-baseweb="select"] {
            font-family: Cambria, serif !important;
            background-color: #393346 !important;
            color: white !important;
            border-radius: 8px !important;
            font-size: 18px !important;
        }

        /* Selected option text */
        div[data-baseweb="select"] span {
            font-family: Cambria, serif !important;
            color: white !important;
            font-size: 18px !important;
        }

        /* Dropdown menu (when open) */
        div[data-baseweb="popover"] {
            font-family: Cambria, serif !important;
            background-color: #393346 !important;
            color: white !important;
            font-size: 18px !important;
        }

        /* Individual option styling */
        div[role="option"] {
            font-family: Cambria, serif !important;
            background-color: #393346 !important;
            color: white !important;
            font-size: 18px !important;
        }

        /* Highlighted option on hover */
        div[role="option"]:hover {
            background-color: #393346 !important;
        }
    </style>
""", unsafe_allow_html=True)

age_group = st.selectbox("",["Select an option", "Kids", "Teens", "Adults"])
questions = []


if age_group != "Select an option":
    questions = questions_by_group[age_group]
    st.markdown("""
        <style>
            .custom-question {
                font-family: 'Playfair Display', serif;
                color: #393346;
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 8px;
                margin-top: 16px;
            }
        </style>
        <h3 style='font-family: "Playfair Display", serif; color: #393346;'>Behavioral Questions</h3>
    """, unsafe_allow_html=True)

    for idx, question in enumerate(questions):
        st.markdown(f"<div class='custom-question'>Q{idx+1}. {question}</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        selected = st.session_state.answers[idx]
        st.markdown("""
    <style>
        div.stButton > button {
            font-family: Cambria, serif !important;
            font-size: 16px !important;
            font-weight: 600;
            padding: 8px 16px;
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

        with col1:
            if st.button("✅ Agree", key=f"agree_{idx}"):
                st.session_state.answers[idx] = 1
        with col2:
            if st.button("❌ Disagree", key=f"disagree_{idx}"):
                st.session_state.answers[idx] = 0

        # Visual feedback
        # Visual feedback
        if st.session_state.answers[idx] == 1:
            st.markdown(f'''
                <p style="font-family: Cambria, serif; color: #393346; background-color: #c0f0c0; padding: 0.3em 1em; border-radius: 6px; display: inline-block;">
                    You selected: Agree ✅
                </p>
            ''', unsafe_allow_html=True)
        elif st.session_state.answers[idx] == 0:
            st.markdown(f'''
                <p style="font-family: Cambria, serif; color: #393346; background-color: #f5baba; padding: 0.3em 1em; border-radius: 6px; display: inline-block;">
                    You selected: Disagree ❌
                </p>
            ''', unsafe_allow_html=True)


st.markdown("""
    <style>
    /* Font styles for dropdown label */
    .stSelectbox label {
        font-family: 'Playfair Display', serif !important;
        color: #393346 !important;
    }

    /* Dropdown background and border */
    div[data-baseweb="select"] {
        background-color: #393346 !important;
        font-family: 'Playfair Display', serif !important;
        border-radius: 8px !important;
    }

    /* Selected option text inside dropdown */
    div[data-baseweb="select"] span {
        color: white !important;
    }

    /* Dropdown options */
    div[role="option"] {
        font-family: 'Playfair Display', serif !important;
        background-color: #393346 !important;
        color: white !important;
    }

    /* Hover effect for dropdown options */
    div[role="option"]:hover {
        background-color: #4b3e59 !important;
    }

    /* Slider label text */
    .stSlider label, .slider-labels span {
        font-family: 'Playfair Display', serif !important;
        color: #393346 !important;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    /* Dropdown container */
    div[data-baseweb="select"] {
        font-family: 'Cambria', serif !important;
        background-color: #393346 !important;
        color: white !important;
        border-radius: 18px !important;
    }

    /* Selected option text */
    div[data-baseweb="select"] span {
        font-family: 'Cambria', serif !important;
        color: white !important;
    }

    /* Dropdown menu */
    div[data-baseweb="popover"] {
        font-family: 'Cambria', serif !important;
        background-color: #393346 !important;
        color: white !important;
    }

    /* Dropdown options */
    div[role="option"] {
        font-family: 'Cambria', serif !important;
        background-color: #393346 !important;
        color: white !important;
    }

    /* On hover */
    div[role="option"]:hover {
        background-color: #393346 !important;
    }

    /* Labels like "Gender", "Jaundice at birth", etc. */
    label.css-17z3815 {
        font-family: 'Playfair Display', serif !important;
        color: white !important;
        font-size: 18px !important;
    }
    </style>
""", unsafe_allow_html=True)



if all(answer is not None for answer in st.session_state.answers):
    st.markdown("""
    <style>
        .select-label {
            font-family: 'Playfair Display', serif;
            font-size: 34px;
            font-weight: bold;
            color: #393346;
            margin-bottom: 30px;
        }
        .spacer {
            height: 2em;
        }
    </style>
    <div class="select-label">User Details</div>
    <div class="spacer"></div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
        .select-label {
            font-family: 'Playfair Display', serif;
            font-size: 24px;
            font-weight: bold;
            color: #393346;
            margin-bottom: 0px;
        }
    </style>
    <div class="select-label">Gender</div>
    """, unsafe_allow_html=True)
    gender = st.selectbox("", ["Select an option", "Female", "Male"])
    st.markdown("""
    <style>
        .select-label {
            font-family: 'Playfair Display', serif;
            font-size: 24px;
            font-weight: bold;
            color: #393346;
            margin-bottom: 0px;
        }
    </style>
    <div class="select-label">Jaundice at birth</div>
    """, unsafe_allow_html=True)
    jaundice = st.selectbox("", ["Select an option", "No", "Yes"])
    st.markdown("""
    <style>
        .select-label {
            font-family: 'Playfair Display', serif;
            font-size: 24px;
            font-weight: bold;
            color: #393346;
            margin-bottom: 0px;
        }
    </style>
    <div class="select-label">Test taken by</div>
    """, unsafe_allow_html=True)
    test_taker = st.selectbox("", ["Select an option", "Parent", "Self", "Caregiver", "Healthcare professional"])
    st.markdown("""
    <style>
        .select-label {
            font-family: 'Playfair Display', serif;
            font-size: 24px;
            font-weight: bold;
            color: #393346;
            margin-bottom: 0px;
        }
    </style>
    <div class="select-label">Ethnicity</div>
    """, unsafe_allow_html=True)
    ethnicity = st.selectbox("", ["Select an option"] + ethnicity_list)
    st.markdown("""
    <style>
        .select-label {
            font-family: 'Playfair Display', serif;
            font-size: 24px;
            font-weight: bold;
            color: #393346;
            margin-bottom: 0px;
        }
    </style>
    <div class="select-label">Country of Residence</div>
    """, unsafe_allow_html=True)
    country = st.selectbox("", ["Select an option"] + country_list)

    if age_group == "Kids":
        st.markdown("""
        <style>
            .select-label {
                font-family: 'Playfair Display', serif;
                font-size: 24px;
                font-weight: bold;
                color: #393346;
                margin-bottom: 0px;
            }
        </style>
        <div class="select-label">Select Age</div>
        """, unsafe_allow_html=True)
        age = st.slider("", min_value=1, max_value=12, step=1)
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
        st.markdown("""
        <style>
            .select-label {
                font-family: 'Playfair Display', serif;
                font-size: 24px;
                font-weight: bold;
                color: #393346;
                margin-bottom: 0px;
            }
        </style>
        <div class="select-label">Select Age</div>
        """, unsafe_allow_html=True)
        age = st.slider("", min_value=13, max_value=19, step=1)
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
        st.markdown("""
        <style>
            .select-label {
                font-family: 'Playfair Display', serif;
                font-size: 24px;
                font-weight: bold;
                color: #393346;
                margin-bottom: 0px;
            }
        </style>
        <div class="select-label">Select Age</div>
        """, unsafe_allow_html=True)
        age = st.slider("", min_value=20, max_value=80, step=1)
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
            for key in ['jaundice_at_birth_no','jaundice_at_birth_yes']:
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
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display&display=swap');

        .custom-info-box {
            background-color: #4ab3fc;
            padding: 18px;
            border-radius: 10px;
            color: white;
            font-size: 20px;
            font-weight: 600;
            font-family: 'Playfair Display', serif;
            text-align: center;
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