import pandas as pd
from joblib import dump, load
import joblib
import re

def process(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["user_id"])
    df_filtered = df[(df['hit'] >= 10) & (df['buyer order'].isna())].copy()

    # === CONFIG ===
    THRESHOLD_TOTAL_USERNAME_BAN = 10
    THRESHOLD_TOTAL_USERNAME_MONITOR = 5

    # === FUNCTION DEFINITIONS ===
    # Convert to integer
    def convert_float_columns_to_integer(df, columns):
        for column in columns:
            df[column] = df[column].astype(int)
        return df

    # Convert to string
    def convert_float_columns_to_string(df, columns):
        for column in columns:
            df[column] = df[column].astype(str)
        return df

    # load model
    def load_model_and_vectorizer(model_path, vectorizer_path):
        return load(model_path), load(vectorizer_path)

    # Pre-processing action (Monitor 5 or Normal before process)
    def pre_processing_action(df):
        df['total_username_preprocessing'] = df.groupby(['registration_channel','registration_ip'])['user_name'].transform('count')
        df['suggest_action'] = df.apply(
            lambda row: 'Normal' if row['total_username_preprocessing'] < THRESHOLD_TOTAL_USERNAME_MONITOR
            else 'Monitor 5' if row['total_username_preprocessing'] < THRESHOLD_TOTAL_USERNAME_BAN
            else 'Checking',
            axis=1
        )
        return df

    # Detect email non google & yahoo
    def annotate_email_non_google_yahoo(df):
        df["suspicious_email"] = df['email'].fillna('None')
        df["suspicious_email"] = df["suspicious_email"].apply(lambda x: "None" if x == "nan" else
                                                            "Yes" if not ("yahoo" in x or "gmail" in x) else "No")
        df["count_suspicious_email"] = df.groupby(['registration_channel','registration_ip'])['suspicious_email'].transform(lambda x: (x == 'Yes').sum())
        df['suggest_action'] = df['suggest_action'].mask(
            (df['suspicious_email'] == "Yes") &
            (df['count_suspicious_email'] >= THRESHOLD_TOTAL_USERNAME_BAN) &
            (df['suggest_action'] == 'Checking'), 'Ban')
        return df

    # Detect pattern email with 3 or more digits
    def annotate_email_3digit(df):
        df['email_3_digit'] = df.apply(
        lambda row: "Yes" if row['suggest_action'] == 'Checking' and re.search(r'[a-zA-Z0-9]+\d{3,}@gmail.com', row['email'])
        else "No" if row['suggest_action'] == 'Checking'
        else "#N/A",
        axis=1
        )
        df["count_email_3_digit"] = df.groupby(['registration_channel','registration_ip'])["email_3_digit"].transform(lambda x: (x == 'Yes').sum())
        df['suggest_action'] = df['suggest_action'].mask(
            (df['email_3_digit'] == "Yes") &
            (df['count_email_3_digit'] >= THRESHOLD_TOTAL_USERNAME_BAN) &
            (df['suggest_action'] == 'Checking'),
            'Ban'
        )
        return df

    # Detect same pattern email
    def annotate_same_pattern_email(df):
        extracted_same_pattern_email = df.loc[df['suggest_action'] == 'Checking', 'email'].str.extract(r'^([a-zA-Z._-]+)(\d+)(@.+)$')
        df.loc[df['suggest_action'] == 'Checking', ['prefix_email', 'number', 'domain']] = extracted_same_pattern_email.values
        df['prefix_email'] = df['prefix_email'].fillna(df['user_name'])
        df['count_same_prefix_email'] = df.groupby(['registration_channel','registration_ip','prefix_email'])['prefix_email'].transform('count')
        df['suggest_action'] = df['suggest_action'].mask(
        (df['suggest_action'] == 'Checking') & (df['count_same_prefix_email'] >= THRESHOLD_TOTAL_USERNAME_BAN),
        'Ban')
        return df

    # Detect patern username with 3 or more digits
    def annotate_username_3digit(df):
        df['username_3digit'] = df.apply(
        lambda row: "Yes" if row['suggest_action'] == 'Checking' and re.search(r'[a-zA-Z]+\d{3,}$', row['user_name'])
        else "No" if row['suggest_action'] == 'Checking'
        else "#N/A",
        axis=1
        )
        df["count_username_3digit"] = df.groupby(['registration_channel','registration_ip'])["username_3digit"].transform(lambda x: (x == 'Yes').sum())
        df['suggest_action'] = df['suggest_action'].mask(
            (df['username_3digit'] == "Yes") &
            (df['count_username_3digit'] >= THRESHOLD_TOTAL_USERNAME_BAN) &
            (df['suggest_action'] == 'Checking'),
            'Ban'
        )
        return df

    # Detect same pattern username
    def annotate_same_pattern_username(df):
        extracted_same_pattern_username = df.loc[df['suggest_action'] == 'Checking', 'user_name'].str.extract(r'^([A-Za-z_@.-][A-Za-z_@.-]*)([0-9]+)$')
        df.loc[df['suggest_action'] == 'Checking', ['prefix_username', 'number_username']] = extracted_same_pattern_username.values
        df['prefix_username'] = df['prefix_username'].fillna(df['user_name'])
        df['count_same_prefix_username'] = df.groupby(['registration_channel','registration_ip','prefix_username'])['prefix_username'].transform('count')
        df['suggest_action'] = df['suggest_action'].mask(
        (df['suggest_action'] == 'Checking') & (df['count_same_prefix_username'] >= THRESHOLD_TOTAL_USERNAME_BAN),
        'Ban')
        return df

    # Post-processing action (Monitor 5 or Normal after process)
    def post_processing_action(df):
        count_username_post_processing = df[df['suggest_action'] == 'Checking'].groupby(['registration_channel', 'registration_ip'])['user_name'].transform('count')
        df.loc[df['suggest_action'] == 'Checking', 'total_username_postprocessing'] = count_username_post_processing
        df['suggest_action'] = df.apply(
                lambda row: 'Normal' if row['total_username_postprocessing'] < THRESHOLD_TOTAL_USERNAME_MONITOR
                else 'Monitor 5' if row['total_username_postprocessing'] < THRESHOLD_TOTAL_USERNAME_BAN
                else row['suggest_action'],
                axis=1
        )
        return df

    # Predict Gibberish username
    def predict_gibberish(df, model, vectorizer):
        df_gib = df[df['suggest_action'] == 'Checking'].copy()
        X_user = vectorizer.transform(df_gib['user_name'])
        df_gib['Gibberish_Predictions'] = model.predict(X_user)
        return df.merge(df_gib[['user_id', 'Gibberish_Predictions']], on='user_id', how='left')

    def finalize_gibberish_counts(df):
        df['Gibberish_Predictions'] = df['Gibberish_Predictions'].astype('Int64')
        df['jumlah_gibberish_groupby'] = df.groupby(['registration_channel', 'registration_ip'])['Gibberish_Predictions'].transform(lambda x: (x == 1).sum())
        return df
    
    # Load Model
    model, vectorizer = load_model_and_vectorizer(
        './models/model_tf.joblib',
        './models/tfidf_vectorizer.joblib'
    )

    # === MAIN FLOW ===
    columns_to_convert_integer = ['user_id']
    columns_to_convert_string = ['user_name', 'email', 'registration_ip']

    df_filtered = convert_float_columns_to_integer(df_filtered, columns_to_convert_integer)
    df_filtered = convert_float_columns_to_string(df_filtered, columns_to_convert_string)
    df_filtered = pre_processing_action(df_filtered)
    df_filtered = annotate_email_non_google_yahoo(df_filtered)
    df_filtered = annotate_email_3digit(df_filtered)
    df_filtered = annotate_same_pattern_email(df_filtered)
    df_filtered = annotate_username_3digit(df_filtered)
    df_filtered = annotate_same_pattern_username(df_filtered)
    df_filtered = post_processing_action(df_filtered)
    df_filtered = predict_gibberish(df_filtered, model, vectorizer)
    df_filtered = finalize_gibberish_counts(df_filtered)

    columns_out = [
    "user_id", "user_name", "registration_ip", "registration_channel", "email",
    "suggest_action", 'prefix_username', 'number_username', "Gibberish_Predictions", "jumlah_gibberish_groupby"
    ]

    df_filtered_fix = df_filtered[columns_out]

    return df_filtered_fix
    
