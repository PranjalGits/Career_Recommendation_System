# Load packages
import streamlit as st
import pickle
import numpy as np

# Load the scaler, label encoder, model, and class names
scaler = pickle.load(open("D:/Career Recommendation System/Models_scaler_2.pkl", 'rb'))
model = pickle.load(open("D:/Career Recommendation System/Models_model_2.pkl", 'rb'))
class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer', 'AI engineer', 'AI Engineer', 'Big Data Engineer', 'Data Scientists', 'Web Developer']

# Recommendations function
def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score, average_score):
    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0

    # Create feature array
    feature_array = np.array([[gender_encoded, part_time_job_encoded, absence_days, extracurricular_activities_encoded,
                               weekly_self_study_hours, math_score, history_score, physics_score,
                               chemistry_score, biology_score, english_score, geography_score, total_score,
                               average_score]])

    # Scale features
    scaled_features = scaler.transform(feature_array)

    # Predict using the model
    probabilities = model.predict_proba(scaled_features)

    # Convert probabilities to percentage format
    probabilities_percentage = [round(prob * 100, 2) for prob in probabilities[0]]

    # Get top five predicted classes along with their probabilities
    top_classes_idx = np.argsort(-probabilities[0])[:5]
    top_classes_names_probs = [(class_names[idx], probabilities_percentage[idx]) for idx in top_classes_idx]

    return top_classes_names_probs

# Streamlit app
def main():
    st.title('Career Recommendation System')

    gender = st.radio('Gender', ('Male', 'Female'))
    part_time_job = st.checkbox('Part Time Job')
    absence_days = st.number_input('Absence Days', min_value=0, max_value=365, step=1)
    extracurricular_activities = st.checkbox('Extracurricular Activities')
    weekly_self_study_hours = st.number_input('Weekly Self Study Hours', min_value=0, max_value=168, step=1)
    math_score = st.number_input('Math Score', min_value=0, max_value=100, step=1)
    history_score = st.number_input('History Score', min_value=0, max_value=100, step=1)
    physics_score = st.number_input('Physics Score', min_value=0, max_value=100, step=1)
    chemistry_score = st.number_input('Chemistry Score', min_value=0, max_value=100, step=1)
    biology_score = st.number_input('Biology Score', min_value=0, max_value=100, step=1)
    english_score = st.number_input('English Score', min_value=0, max_value=100, step=1)
    geography_score = st.number_input('Geography Score', min_value=0, max_value=100, step=1)
    # Calculate total score and average score
    total_score = math_score + history_score + physics_score + chemistry_score + biology_score + english_score + geography_score
    average_score = total_score / 7  # Assuming there are 7 subjects

    st.write(f"Total Score: {total_score}")
    st.write(f"Average Score: {average_score}")

    if st.button('Recommend'):
        recommendations = Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                                          weekly_self_study_hours, math_score, history_score, physics_score,
                                          chemistry_score, biology_score, english_score, geography_score,
                                          total_score, average_score)
        st.write("Top recommended careers:")
        for recommendation in recommendations:
            st.write(f"- {recommendation[0]} with probability {recommendation[1]:.2f}")

if __name__ == '__main__':
    main()
