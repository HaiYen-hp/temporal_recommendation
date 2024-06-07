import pandas as pd
import math
Ks = [30,50]
def evaluate(model, va_dataset, num_candidates, sequence_length):
    HR30 = 0
    HR50 = 0
    Recall30 = 0
    Recall50 = 0
    NDCG30 = 0
    NDCG50 = 0
    num_users = int(len(va_dataset) / num_candidates)

    for i in range(num_users):
        va_batch = va_dataset.iloc[i * num_candidates : (i + 1) * num_candidates]

        user_input = va_batch.user_id
        item_input = va_batch.item_id

        recent_month_inputs = []
        recent_day_inputs = []
        recent_date_inputs = []
        recent_hour_inputs = []
        recent_timestamp_inputs = []
        recent_itemid_inputs = []

        month_input = va_batch.month
        day_input = va_batch.day_of_week
        date_input = va_batch.date
        hour_input = va_batch.hour
        timestamp_input = va_batch.timestamp
        for j in range(sequence_length):
            recent_month_inputs.append(va_batch['month' + str(j)])
            recent_day_inputs.append(va_batch['day_of_week' + str(j)])
            recent_date_inputs.append(va_batch['date' + str(j)])
            recent_hour_inputs.append(va_batch['hour' + str(j)])
            recent_timestamp_inputs.append(va_batch['timestamp' + str(j)])
            recent_itemid_inputs.append(va_batch['item_id' + str(j)])
        labels = va_batch.rating

        prob = pd.DataFrame(model.predict([user_input, item_input, month_input, day_input, date_input, hour_input, timestamp_input] + [recent_month_inputs[j] for j in range(sequence_length)]+ [recent_day_inputs[j] for j in range(sequence_length)]+ [recent_date_inputs[j] for j in range(sequence_length)]+ [recent_hour_inputs[j] for j in range(sequence_length)]+ [recent_timestamp_inputs[j] for j in range(sequence_length)] + [recent_itemid_inputs[j] for j in range(sequence_length)], batch_size=len(va_batch)), columns=['prob'])
        
        va_batch = (va_batch.reset_index(drop=True)).join(prob)
        # print(f'len_top30: {len(top30)}')
        # print(f'len_item_input {len(item_input)}')
        # print(item_input)
        metric_Ks = {}
        for k in Ks:    
            metric_Ks[f'top{k}'] = va_batch.nlargest(k, 'prob')
            metric_Ks[f'hit{k}'] = int(1 in metric_Ks[f'top{k}'].rating.tolist())
            # metric_Ks[f'intersect{k}'] = len(set(item_input[:k]) & set(metric_Ks[f'top{k}'].item_id))
            # metric_Ks[f'recall{k}'] = metric_Ks[f'intersect{k}']/len(item_input[:k])        
            if metric_Ks[f'hit{k}']:
                metric_Ks[f'intersect{k}'] = len(set(item_input[:k]) & set(metric_Ks[f'top{k}'].item_id))
                metric_Ks[f'recall{k}'] = metric_Ks[f'intersect{k}']/len(item_input[:k])
                ind = metric_Ks[f'top{k}'].rating.tolist().index(1) + 1
                metric_Ks[f'ndcg{k}'] = float(1) / math.log(float(ind + 1), 2)
            else:
                metric_Ks[f'recall{k}'] = 0
                metric_Ks[f'ndcg{k}'] = 0
            

        # print(metric_Ks.keys())
        HR30 += metric_Ks['hit30']
        HR50 += metric_Ks['hit50']
        Recall30 += metric_Ks['recall30']
        Recall50 += metric_Ks['recall50']
        NDCG30 += metric_Ks['ndcg30']
        NDCG50 += metric_Ks['ndcg50']

        va_batch.drop(columns=['prob'], inplace=True)

    HR30 = float(HR30) / num_users
    HR50 = float(HR50) / num_users
    Recall30 = Recall30 / num_users
    Recall50 = Recall50 / num_users
    NDCG30 = NDCG30 / num_users
    NDCG50 = NDCG50 / num_users

    return HR30, HR50, Recall30, Recall50, NDCG30, NDCG50