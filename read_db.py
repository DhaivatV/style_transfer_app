import sqlite3
import pandas as pd
import pickle
import  base64
import os

# Create your connection.
def get_result(taskId):
    original_object = []
    cnx = sqlite3.connect('db.sqlite3')
    df = pd.read_sql_query("Select * FROM celery_taskmeta", cnx)
    # taskid  = input()
    row = df. loc[df['task_id'] == taskId.strip()]
    if (row['status'].iloc[0]) == 'SUCCESS':
        res = row['result']
        value = res.iloc[0]
        blob = value
        val = pickle.loads(blob)
       
        # print(path)
        # with open(path, 'rb') as image_file:
        #     encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        #     original_object.append(encoded_string)
        # # original_object.append(pickle.loads(blob))
        cnx.commit()
        cnx.close()


        return val

    else :
        return (row['status'].iloc[0])
    
def check_status(taskId):
    cnx = sqlite3.connect('db.sqlite3')
    df = pd.read_sql_query("Select * FROM celery_taskmeta", cnx)
    # taskid  = input()
    row = df. loc[df['task_id'] == taskId.strip()]
    return row['status'].iloc[0]


    
# print(os.path.exists("D:\\celery_demo\\images\\blend-weighted.png"))
# print(get_result('b472b513-76d0-44c7-890c-8363d6a78cec'))
# D:\celery_demo\style_transfer\blend-weighted.png