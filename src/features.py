def serialize_student_row(row):
    """
    Converts a structured student record into a natural language narrative.
    
    Input: Row (Series) -> {'Age': 20, 'CGPA': 3.5, 'Depression': 1,...}
    Output: String -> "A 20-year-old student reports..."
    """
    # 1. Map numerical codes to descriptive text if necessary
    sleep = str(row.get('Sleep Duration', 'unknown'))
    pressure = row.get('Academic Pressure', 'unknown')
    diet = row.get('Dietary Habits', 'unknown')
    
    # 2. Construct the Narrative Template
    # We vary the template slightly to make the model robust (optional)
    text = (
        f"The subject is a {int(row['Age'])}-year-old student. "
        f"They have a CGPA of {row['CGPA']} and report an academic pressure level of {pressure} out of 5. "
        f"Their sleep duration is described as '{sleep}' and they have {diet} dietary habits. "
        f"Work/Study hours per day: {row}. "
        f"Family history of mental illness: {row['Family History of Mental Illness']}."
    )
    
    return text



def prepare_hybrid_dataset(student_df, reddit_df):
    """
    Merges the two datasets into a single Text + Label format.
    """
    # Convert Student Data
    student_df['text_input'] = student_df.apply(serialize_student_row, axis=1)
    student_df = student_df.rename(columns={'Depression': 'label'}) # Target variable
    
    # Prepare Reddit Data
    reddit_df = reddit_df.rename(columns={'full_text': 'text_input'})
    
    # Standardize Labels (Ensure both use 0/1 for Healthy/At-Risk)
    # (Assuming you have mapped Reddit subreddits to 0/1 previously)
    
    return pd.concat([
        student_df[['text_input', 'label']], 
        reddit_df[['text_input', 'label']], axis=0).sample(frac=1) # Shuffle