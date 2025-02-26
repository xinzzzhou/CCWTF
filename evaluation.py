import pandas as pd
import numpy as np
from _metric_ import metric
import os
import re


def parse_result(df, h):
    trues = []
    preds = []

    for row in df.iterrows():
        if ']' not in row[1]['Realvalues']:
            continue
        true = row[1]['Realvalues'].split(']')[0].split('[')[1]
        pred = row[1]['Results'].split(']<end_predicted_value>,')[0].split('<begin_predicted_value>[')[1].replace(" ", "")

        true = [int(num) for num in re.findall(r'\b\d+\b', true)]
        pred = [int(num) for num in re.findall(r'\b\d+\b', pred)][:h]

        trues.append(true)
        preds.append(pred)

    return trues, preds


# get all the files in the llm_result folder
all_file_paths = os.listdir('llm_result')
output_path = 'eva_metric.csv'
for file_path in all_file_paths:
    print(file_path)
    
    if 'gpt' not in file_path and 'tfidf' not in file_path:
        continue
    h = int(file_path.split('_')[2])
    k = int(file_path.split('_')[-4])
    df = pd.read_csv(f'llm_result/{file_path}')
    trues, preds = parse_result(df, h)
    if file_path == 'zero-shot_(\'gpt-4o-mini-2024-07-18\',)_1_summaryandtitle_llama_1B_8_forecast_result_b.csv':
        print('here')

    trues_np = np.array(trues)
    preds_np = np.array(preds)
    rmse, wrmspe, mae, wape, mse, smape, mape, mspe = metric(trues_np, preds_np)

    # log the results, each time add a new row
    with open(output_path, 'a') as f:
        f.write(f'{file_path}, {h}, {k}, {rmse},{wrmspe},{mae},{wape},{mse},{smape},{mape},{mspe}\n')
