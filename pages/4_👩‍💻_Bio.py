import streamlit as st

st.set_page_config(page_title='About Dev - NeuroNet MRI', 
                    page_icon='👩‍💻',
                    layout="wide")
st.title("About the Developer - NeuroNet MRI")

col1, col2 = st.columns(2)

col1.subheader("Mikayla Norton")

col1.markdown("###### _noun_")
col1.text("DATA SCIENTIST, CURRENTLY AT INTEL")
col1.write("")

col1.markdown("###### _verb_")
col1.text("1. INNOVATE AND DESIGN")
col1.text("2. INFLUENCE AND ENERGIZE")
col1.write("")


col1.markdown("###### _adjective_")
col1.text("1. PASSIONATE")
col1.text("2. PRECISE")
col1.text("3. GOAL-ORIENTED")
col1.write("")

col1.markdown('<h4 style="text-align: left;">Who am I?</h4>', unsafe_allow_html=True)
col1.markdown("<p style='text-align: justify;'>Nice to 'e-meet' you! My name is Mikayla, I am the developer of NeuroNet MRI. I am a second-year MS student in Data Science. I am a passion-centric learner, with a drive to always challenge myself for something new. I hope you've enjoyed my application so far.</p>", unsafe_allow_html=True)
col1.write("")


col2.image("assets/headshot.jpg")

col1, col2 = st.columns(2)
col1.image("assets/hydro.png", width = 400)
col2.markdown('<h3 style="text-align: left;">Motivation</h3>', unsafe_allow_html=True)
col2.markdown('<p style="text-align: justify;">My journey towards developing a CNN-based tumor classifier for MRI scans is deeply rooted in my personal battle with hydrocephalus, a brain disorder that has profoundly impacted my life. This experience has not only given me a unique insight into the challenges faced by individuals with neurological conditions but also ignited a burning passion to leverage the power of technology for medical advancements. Recognizing the critical role of early and accurate tumor detection in improving patient outcomes, I was inspired to embark on this project. </p>', unsafe_allow_html=True)
col2.markdown("<p style='text-align: justify;'>My aim is to harness the capabilities of Convolutional Neural Networks to analyze MRI scans with greater precision and efficiency. By doing so, I aspire to contribute to a world where the diagnosis of brain tumors is not only swift but also more accurate, reducing the uncertainty and anxiety experienced by patients. This project is not just a pursuit of technological innovation; it's a personal crusade to transform the landscape of neurological healthcare, driven by my own experiences and challenges with hydrocephalus.</p>", unsafe_allow_html=True)


