# processors/bulk_full_ip.py
import pandas as pd
import re
from joblib import load
import joblib

# Fungsi untuk memproses data CSV yang di-upload dan mengembalikan hasil
def process(df: pd.DataFrame) -> pd.DataFrame:
    df_filtered = df[df["sum_buyer_mp_placed_order_cnt_1d"].isna()].copy()
    # === CONFIG ===
    THRESHOLD_TOTAL_USERNAME_BAN = 10
    THRESHOLD_TOTAL_USERNAME_MONITOR = 5

    # === FUNCTION DEFINITIONS ===

    # Convert float to object
    def convert_float_columns_to_object(df, columns):
        for column in columns:
            df[column] = df[column].astype('object')
        return df

    # load model
    def load_model_and_vectorizer(model_path, vectorizer_path):
        return load(model_path), load(vectorizer_path)

    # Ban szdf
    def ban_szdf(df):
        df.loc[
            df['count szdf'] >= 10,
            ['to check','action']] = ['szdf','Ban']
        return df

    # Ban Hit >= 20
    def ban_hit_20(df):
        df.loc[
            (df['to check'].isna()) &
            (df['action'].isna()) &
            (df['hit'] >= 20),
            ['to check','action']] = ['IP 20','Ban']
        return df

    # to check IP
    def check_ip(df):
        df.loc[
            (df['to check'].isna()) &
            (df['action'].isna()) &
            (df['hit'] >= 10),
            ['to check','action']] = ['IP','Checking']
        return df

    # Pre-processing action (Monitor 5 or Normal before process)
    def pre_processing_action(df):
        count_username_pre_processing = df[df['action'] == 'Checking'].groupby(['registration_method', 'registration_ip'])['user_name'].transform('count')
        df.loc[df['action'] == 'Checking', 'total_username_postprocessing'] = count_username_pre_processing
        df['action'] = df.apply(
            lambda row: 'Normal' if row['total_username_postprocessing'] < THRESHOLD_TOTAL_USERNAME_MONITOR
            else 'Monitor 5' if row['total_username_postprocessing'] < THRESHOLD_TOTAL_USERNAME_BAN
            else row['action'],
            axis=1
        )
        return df

    # Detect patern username with 3 or more digits
    def annotate_username_3digit(df):
        df.loc[
            (df['user_name'].str.contains(r'^[a-zA-Z]+\d{3,}$')) &
            (df['action'] == 'Checking') &
            (df['to check'] == 'IP'),
            'Username_3_digit'] = 'Yes'
        df["Banyak_Username_3_digit"] = df.groupby(['registration_method','registration_ip'])["Username_3_digit"].transform(lambda x: (x == 'Yes').sum())
        df.loc[
            (df['Banyak_Username_3_digit'] >= THRESHOLD_TOTAL_USERNAME_BAN) &
            (df['Username_3_digit'] == "Yes") &
            (df['action'] == 'Checking') &
            (df['to check'] == 'IP'), 'action'] = 'Ban'
        return df

    # Detect same pattern username
    def annotate_same_pattern_username(df):
        extracted_same_pattern_username = df.loc[df['action'] == 'Checking','user_name'].str.extract(r'^([A-Za-z_@.-][A-Za-z_@.-]*)([0-9]+)$')
        df.loc[df['action'] == 'Checking', ['prefix_username', 'number_usernamme']] = extracted_same_pattern_username.values
        df['prefix_username'] = df['prefix_username'].fillna(df['user_name'])
        df['count_same_prefix_username'] = df.groupby(['registration_method','registration_ip','prefix_username'])['prefix_username'].transform('count')
        df['action'] = df['action'].mask(
        (df['action'] == 'Checking') & (df['count_same_prefix_username'] >= THRESHOLD_TOTAL_USERNAME_BAN),
        'Ban')
        return df

    # Post-processing action (Monitor 5 or Normal after process)
    def post_processing_action(df):
        count_username_post_processing = df[df['action'] == 'Checking'].groupby(['registration_method', 'registration_ip'])['user_name'].transform('count')
        df.loc[df['action'] == 'Checking', 'total_username_postprocessing'] = count_username_post_processing
        df['action'] = df.apply(
                lambda row: 'Normal' if row['total_username_postprocessing'] < THRESHOLD_TOTAL_USERNAME_MONITOR
                else 'Monitor 5' if row['total_username_postprocessing'] < THRESHOLD_TOTAL_USERNAME_BAN
                else row['action'],
                axis=1
        )
        return df

    # Predict Gibberish username
    def predict_gibberish(df, model, vectorizer):
        df_gib = df[(df['action'] == 'Checking') & (df['to check'] == 'IP')].copy()
        X_user = vectorizer.transform(df_gib['user_name'])
        df_gib['Gibberish_Predictions'] = model.predict(X_user)
        return df.merge(df_gib[['user_id', 'Gibberish_Predictions']], on='user_id', how='left')

    def finalize_gibberish_counts(df):
        df['Gibberish_Predictions'] = df['Gibberish_Predictions'].astype('Int64')
        df['jumlah_gibberish_groupby'] = df.groupby(['registration_method', 'registration_ip'])['Gibberish_Predictions'].transform(lambda x: (x == 1).sum())
        return df

    # Load Model
    model, vectorizer = load_model_and_vectorizer(
        './models/model_tf.joblib',
        './models/tfidf_vectorizer.joblib'
    )

    # === MAIN FLOW ===
    columns_to_convert = ['to check', 'action']
    df_filtered = convert_float_columns_to_object(df_filtered, columns_to_convert)
    df_filtered = ban_szdf(df_filtered)
    df_filtered = ban_hit_20(df_filtered)
    df_filtered = check_ip(df_filtered)
    df_filtered = pre_processing_action(df_filtered)
    df_filtered = annotate_username_3digit(df_filtered)
    df_filtered = annotate_same_pattern_username(df_filtered)
    df_filtered = post_processing_action(df_filtered)
    df_filtered = predict_gibberish(df_filtered, model, vectorizer)
    df_filtered = finalize_gibberish_counts(df_filtered)

    columns_out = [
        "user_id", "registration_ip", "user_name", "registration_method", "hit",
        "to check", "action", "Gibberish_Predictions", "jumlah_gibberish_groupby"
    ]

    df_filtered_fix = df_filtered[columns_out]


    return df_filtered_fix

