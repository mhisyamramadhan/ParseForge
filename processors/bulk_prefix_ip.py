# processors/bulk_prefix_ip.py
import pandas as pd
from joblib import dump, load
import joblib
import re

def process(df: pd.DataFrame) -> pd.DataFrame:
    df_filtered = df[(df['hit'] >= 10) & (df['sum_buyer_mp_placed_order_cnt_1d'].isna())].copy()
    # === CONFIG ===
    THRESHOLD_TOTAL_USERNAME_BAN = 10
    THRESHOLD_TOTAL_USERNAME_MONITOR = 5

    # === FUNCTION DEFINITIONS ===

    # load model
    def load_model_and_vectorizer(model_path, vectorizer_path):
        return load(model_path), load(vectorizer_path)

    # Pre-processing action (Monitor 5 or Normal before process)
    def pre_processing_action(df):
        df['total_username_preprocessing'] = df.groupby(['registration_method', 'prefix'])['user_name'].transform('count')
        df['suggest_action'] = df.apply(
            lambda row: 'Normal' if row['total_username_preprocessing'] < THRESHOLD_TOTAL_USERNAME_MONITOR
            else 'Monitor 5'if row['total_username_preprocessing'] < THRESHOLD_TOTAL_USERNAME_BAN
            else 'Checking',
            axis=1
        )
        df = df.drop(columns=['total_username_preprocessing'])
        return df

    # extract pattern username & count same pattern username
    def annotate_same_pattern_username(df):
        extracted_same_pattern_username = df.loc[df['suggest_action'] == 'Checking', 'user_name'].str.extract(r'^([A-Za-z_@.-][A-Za-z_@.-]*)([0-9]+)$')
        df.loc[df['suggest_action'] == 'Checking', ['prefix_username', 'number']] = extracted_same_pattern_username.values
        df['prefix_username'] = df['prefix_username'].fillna(df['user_name'])
        df['jumlah_pola_sama'] = df.groupby(['registration_method', 'prefix', 'prefix_username'])['prefix_username'].transform('count')
        df['suggest_action'] = df['suggest_action'].mask(
        (df['suggest_action'] == 'Checking') & (df['jumlah_pola_sama'] >= THRESHOLD_TOTAL_USERNAME_BAN),
        'Ban')
        return df

    def annotate_username_3digit(df):
        df['Username_3_digit'] = df.apply(
            lambda row: "Yes" if row['suggest_action'] == 'Checking' and re.search(r'^[a-zA-Z]+\d{3,}$', row['user_name'])
            else "No" if row['suggest_action'] == 'Checking'
            else "#N/A",
            axis=1
        )
        df["Banyak_Username_3_digit"] = df.groupby(['registration_method','prefix'])["Username_3_digit"].transform(lambda x: (x == 'Yes').sum())
        df['suggest_action'] = df['suggest_action'].mask(
            (df['Username_3_digit'] == "Yes") &
            (df['Banyak_Username_3_digit'] >= THRESHOLD_TOTAL_USERNAME_BAN) &
            (df['suggest_action'] == 'Checking'),
            'Ban'
        )
        return df

    # Post-processing action (Monitor 5 or Normal after process)
    def post_processing_action(df):
        count_username = df[df['suggest_action'] == 'Checking'].groupby(['registration_method', 'prefix'])['user_name'].transform('count')
        df.loc[df['suggest_action'] == 'Checking', 'total_username_postprocessing'] = count_username
        df['total_username_postprocessing'] = df['total_username_postprocessing'].fillna(0).astype(int)
        df['suggest_action'] = df['suggest_action'].mask(
            (df['suggest_action'] == "Checking") &
            (df['total_username_postprocessing'] < THRESHOLD_TOTAL_USERNAME_MONITOR),
            'Normal'
        )
        df['suggest_action'] = df['suggest_action'].mask(
            (df['suggest_action'] == "Checking") &
            (df['total_username_postprocessing'] < THRESHOLD_TOTAL_USERNAME_BAN),
            'Monitor 5'
        )
        return df

    # Predict Gibberish username
    def predict_gibberish(df, model, vectorizer):
        df_gib = df[(df['suggest_action'] == 'Checking') & (df['jumlah_pola_sama'] < 2)].copy()
        X_user = vectorizer.transform(df_gib['user_name'])
        df_gib['Gibberish_Predictions'] = model.predict(X_user)
        return df.merge(df_gib[['user_id', 'Gibberish_Predictions']], on='user_id', how='left')

    def finalize_gibberish_counts(df):
        df['Gibberish_Predictions'] = df['Gibberish_Predictions'].astype('Int64')
        df['jumlah_gibberish_groupby'] = df.groupby(['registration_method', 'prefix'])['Gibberish_Predictions'].transform(lambda x: (x == 1).sum())
        return df

    # Load Model
    model, vectorizer = load_model_and_vectorizer(
        './models/model_tf.joblib',
        './models/tfidf_vectorizer.joblib'
    )

    # === MAIN FLOW ===
    df_filtered = pre_processing_action(df_filtered)
    df_filtered = annotate_same_pattern_username(df_filtered)
    df_filtered = annotate_username_3digit(df_filtered)
    df_filtered = post_processing_action(df_filtered)
    df_filtered = predict_gibberish(df_filtered, model, vectorizer)
    df_filtered = finalize_gibberish_counts(df_filtered)

    columns_out = [
    "user_id", "prefix", "user_name", "prefix_username", "number", "jumlah_pola_sama",
    "total_username_postprocessing", "suggest_action", "Gibberish_Predictions", "jumlah_gibberish_groupby"
    ]

    df_filtered_fix = df_filtered[columns_out]

    return df_filtered_fix