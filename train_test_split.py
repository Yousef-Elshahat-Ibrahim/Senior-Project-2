import os
import pandas as pd
from sklearn.model_selection import train_test_split

dir = f'{os.getcwd().replace("\\", "/")}/Data/Mallorn Data corrected'
df = pd.read_csv(f"{dir}/big_daddy_train.csv")

A_lambda = [c for c in df.columns if "A_lambda" in c]
dropped =["split","EBV","Z"] + A_lambda
df.drop(columns=dropped+["target"], inplace=True)
cols = df.drop(columns=["object_id","Time (MJD)"]).columns.tolist()

def stratified_unique_split(df, target_col, object_id_col, test_size=0.2, random_state=42):
    unique_objs = df[[object_id_col, target_col]].drop_duplicates(subset=[object_id_col])

    train_objs, test_objs = train_test_split(
        unique_objs,
        test_size=test_size,
        stratify=unique_objs[target_col],
        random_state=random_state
    )

    train_df = df[df[object_id_col].isin(train_objs[object_id_col])]
    test_df = df[df[object_id_col].isin(test_objs[object_id_col])]

    return train_df, test_df

train_df, test_df = stratified_unique_split(df, target_col="SpecType", object_id_col="object_id", test_size=0.2)

train_df.to_csv(f"{dir}/train_df.csv", index=False)
test_df.to_csv(f"{dir}/test_df.csv", index=False)
print("done")