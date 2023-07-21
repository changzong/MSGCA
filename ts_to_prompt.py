import pickle
from datetime import datetime
import pdb

prompt_template = '''From %start_date% to %end_date%, the close prices of stock %stock_name% were %price_ts% on each day.
With the information above, what is the close price going to be on %predict_date%, %week_day%?'''

prompt_template_desc = '''%stock_name% is the stock code of %company_name%. It is a %company_desc%.
From %start_date% to %end_date%, the close prices of stock %stock_name% were %price_ts% on each day.
With the information above, what is the close price going to be on %predict_date%, %week_day%?'''

weekday = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

def data_load(price_file_path):
    with open(price_file_path, 'rb') as f:
        stock_price = pickle.load(f)

    stock_price_by_name = {}
    stock_names = []
    for item in stock_price:
        if item[0] not in stock_price_by_name:
            stock_price_by_name[item[0]] = [[item[1], round(item[2], 3)]]
            stock_names.append(item[0])
        else:
            stock_price_by_name[item[0]].append([item[1], round(item[2], 3)])
    return stock_price_by_name, stock_names

def indicator_ts_to_text_prompt(price_file_path, train_ts_len, idxs):
    stock_price_dict, stock_names = data_load(price_file_path)
    stock_prompts = []
    for k,v in stock_price_dict.items():
        stock_name = k
        start_date = v[0][0]
        end_date = v[train_ts_len][0]
        price_ts_list = [str(item[1]) for item in v]
        price_ts = ','.join(price_ts_list)
        predict_date = v[train_ts_len+1][0]
        week_day = weekday[datetime.strptime(predict_date, '%Y-%m-%d').weekday()]

        prompt = prompt_template.replace('%start_date%', start_date)
        prompt = prompt.replace('%end_date%', end_date)
        prompt = prompt.replace('%stock_name%', stock_name)
        prompt = prompt.replace('%price_ts%', price_ts)
        prompt = prompt.replace('%predict_date%', predict_date)
        prompt = prompt.replace('%week_day%', week_day)

        stock_prompts.append(prompt)

    stock_prompts_sample = []
    for idx in idxs:
        stock_prompts_sample.append(stock_prompts[idx])
    return stock_prompts_sample


if __name__ == '__main__':
    idxs = [40, 7, 1, 17, 15, 14, 8, 6, 34, 5, 37, 27, 2, 47, 49, 13, 44, 32, 36, 46, 42, 22, 20, 28, 30, 41, 48, 33, 18, 43, 0, 35, 24, 10, 38, 39, 3, 12, 21, 31]
    prompts = indicator_ts_to_text_prompt('./data/bd22_stock/stock_51_price.pkl', 252, idxs)
    pdb.set_trace()




