'''
cd data/MultimodalTSF/ArticleTraffic/Wiki-popular/
nohup python zero-shot_gpt4o.py > zero-shot_gpt4o.out 2>&1 &
'''

import pandas as pd
import time
import csv
from openai import OpenAI
import pickle
import numpy as np

import os

from call_llm import call_openai
from prompt_rep import  __zero_shot_head__, get_prompt
from _metric_ import metric
 



def zero_shot_with_gpt4o(end_date, h, page, x_summary, related_text, related_series, exp_type, max_tokens, temperature):


    # Use format method to fill in the placeholders
    instant_prompt = get_prompt(exp_type, h, end_date, page, x_summary, related_text, related_series, similar_method)
    # instant_prompt = __zero_shot_prompt__.format(exp_type=exp_type, h=h, end_date=end_date, title=title, text=text)
    messages=[
            {"role": "system", "content": __zero_shot_head__},
            {"role": "user", "content": instant_prompt}
        ]
#  llama
    result = call_openai(messages, max_tokens, temperature,model="gpt-4o-mini-2024-07-18")

    return result

def result_gpt4o_process(result_text_gpt4o):
    lines = result_text_gpt4o.split('\n')
    forecast_values = []
    explanations = []
    
    for i in range(0, len(lines), 2):
        if i+1 < len(lines):
            day_line = lines[i]
            explanation_line = lines[i+1]
            
            # Extract date and predicted value
            date, value = day_line.split(': ')
            forecast_values.append(int(value))
            
            # Extract explanation (remove "Explanation: " prefix)
            explanation = explanation_line.replace("Explanation: ", "").strip()
            explanations.append(explanation)
    return forecast_values, *explanations



def forecast(ys, pages, x_summaries, related_text_list, related_series_list, exp_type, h, output_file, end_date):
    with open(output_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if csvfile.tell() == 0:
            writer.writerow(['Page', 'Realvalues', 'Results'])  # Header row
            processed_articles = set()
        else:    
            df = pd.read_csv(output_file)
            processed_articles = set(df['Page'].values.tolist())

        for i, page in enumerate(pages):
            if page in processed_articles:
                print(f"Page {page} already processed, skipping...")
                continue
            t1 = time.time()
            # get the corresponding data
            y = ys[i]
            x_summary = x_summaries[i]
            if exp_type != 'summaryonly':
                related_text = related_text_list[i]
                related_series = related_series_list[i]
            else:
                related_text, related_series = [], []
            # find the traffic in df_series of related_articles
            results = zero_shot_with_gpt4o(end_date, h, page, x_summary, related_text, related_series, exp_type, max_tokens, temperature)
            t2 = time.time()
            print(results)
            print(f"Total time taken:{t2-t1}s, GPT-4o Result for sample {i+1} saved to CSV")
            print("-" * 50)  # Separator between results
            writer.writerow([page, y, results])  
    print(f"All results saved to {output_file}")


if __name__ == "__main__":


    from config import __raw_data_path__ 
    from config import __save_data__
    from config import __llm_result__

    setting = 'zero-shot'
    model="gpt-4o-mini-2024-07-18",
    exp_type = 'summaryonly' 
    similar_method = 'tfidf'
    max_tokens = 13000
    temperature = 0

    df_series_test = pd.read_csv(f'{__raw_data_path__}/traffic_daily_final_sample.csv')
    df_series_full = pd.read_csv(f'{__raw_data_path__}/traffic_daily_final.csv')
    df_text_similarity_idx = np.load(f'{__save_data__}/sorted_indices_{similar_method}.npy')
    df_text = pd.read_csv(f'{__raw_data_path__}/traffic_daily_text_final.csv')
    print("Data loaded successfully")
    print(f'traffic_daily_final: {df_series_full}')
    print(f'sorted_indices_{similar_method}: {df_text_similarity_idx}')
    print(f'traffic_daily_text_final: {df_text}')

    for h in [1,2,4,7]:
        for k in [1,2,4,8,16]:
            
            end_date = f'2019/07/0{7+h}'
            output_file = f"{__llm_result__}/{setting}_{model}_{h}_{exp_type}_{similar_method}_{k}_forecast_result_b.csv"

            # get y ground truth, keep the previous 7 days
            y = df_series_test.iloc[:, 10:10+h].values
            print(f'y: {y}')
            
            # get x summary (needed by all exp_type)
            pages = df_series_test['Page'].values
            x_summary = []
            for page in pages:
                x_summary.append(df_text[df_text['Page'] == page]['text'].values[0])
            print(f'x_summary: {x_summary}')

            if exp_type == 'summaryonly':
                related_text_list, related_series_list = [], []
            else:
                # get x related objects (needed by summaryandtitle, summaryandseries, all)
                related_row_list = []
                for text_similarity_idx in df_text_similarity_idx:
                    related_articles = df_series_full.iloc[text_similarity_idx,:]
                    related_row_list.append(related_articles)
                print(f'related_articles_list: {related_row_list}')
                
                # get x related text and related series (needed by summaryandtitle, summaryandseries, all)
                related_text_list = []
                related_series_list = []
                for related_row in related_row_list:
                    top_k_related_row = related_row.iloc[:k,:]
                    related_text = top_k_related_row['Page'].values
                    related_text_list.append(related_text)
                    related_series = top_k_related_row.iloc[:, 3:10].values
                    related_series_list.append(related_series)
                print(f'related_text_list: {related_text_list}')      # 727, k
                print(f'related_series_list: {related_series_list}')  # 727, k, 7  
            
            # Call for GPT-4o   
            forecast(y, pages, x_summary, related_text_list, related_series_list, exp_type, h, output_file, end_date)