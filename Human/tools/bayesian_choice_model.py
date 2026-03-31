# bayesian_choice_model.py (replaced with multinomial logistic model version)

import pandas as pd
import statsmodels.api as sm

def prepare_multinomial_logit_data(df):
    df = df.copy()
    df['prev_choice_same'] = (df['arm'] == df['prev_choice']).astype(int)
    df['win_stay'] = df['prev_choice_same'] * df['prev_reward']
    df['condition_social'] = (df['condition'] == 'social').astype(int)
    return df

def fit_multinomial_logit(df):
    df = prepare_multinomial_logit_data(df)

    # Keep only chosen-arm rows to form y
    df_choice = df[df['choice'] == 1].copy()
    y = df_choice['arm'].astype(int)
    X = df_choice[['own_evidence', 'social_evidence', 'prev_choice_same', 'win_stay', 'condition_social']]
    X = sm.add_constant(X)

    model = sm.MNLogit(y, X)
    result = model.fit(disp=False)
    return result

# Example usage:
# df = pd.read_csv("model_xxx.csv")
# result = fit_multinomial_logit(df)
# print(result.summary())