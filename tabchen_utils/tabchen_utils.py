from common_library import *

## 
def getAbsTime(item):
    item = int(item)
    item=item//1000
    return"{:0>2}:{:0>2}:{:0>2}".format(item//3600,int(item//60%60),int(item%60))

def add_df_start_end_time(df,start_column,end_column):
    df['开始时间'] = df[start_column].apply(getAbsTime)
    df['结束时间'] = df[end_column].apply(getAbsTime)
    return df
    
## mysql

class myMysql:
    def __init__(self, dbhost, dbport, dbname, dbuser, dbpassword):
        self.dbhost = dbhost
        self.dbport = dbport
        self.dbname = dbname
        self.dbuser = dbuser
        self.dbpassword = dbpassword

    def connect(self):
        self.connection = pymysql.connect(host=self.dbhost,
                                          user=self.dbuser,
                                          port=self.dbport,
                                          password=self.dbpassword,
                                          db=self.dbname,
                                          charset='utf8',
                                          cursorclass=pymysql.cursors.DictCursor)
        self.cursor = self.connection.cursor()
        self.connection.autocommit(True)

    def insert(self, table_name, data):
        self.connect()
        items = list(data.keys())
        keys = ",".join(items)
        valueList = list(data.values())
        values = str(valueList)[1:-1]
        sql = "replace into {} ({}) values ({})".format(
            table_name, keys, values)
        self.cursor.execute(sql)
        self.close()

    def execute(self, sql):
        self.connect()
        self.cursor.execute(sql)
        result = self.cursor.fetchall()
        self.close()
        return result

    def close(self):
        self.connection.close()

## model_help
def getModelInfo(y_true,y_pred,show=False):
    try:
        auc=roc_auc_score(y_true, y_pred)
    except:
        auc=-1
    y_pred = y_pred>0.5
    recall=recall_score(y_true, y_pred) 
    precision=precision_score(y_true,y_pred)
    f1=f1_score(y_true, y_pred)
    accuracy=accuracy_score(y_true,y_pred)
    if show:
        for name,value in zip(('Accuracy','Precision','Recall','F_meansure','AUC_Value'),
                              (accuracy,precision,recall,f1,auc)):
            print('{} : {:.4f}'.format(name,value))
            
    report = {'Accuracy':round(accuracy,4),
          'Precision':round(precision,4),
          'Recall':round(recall,4),
          'F_meansure':round(f1,4),
          'AUC_Value':round(auc,4),
         }
    return report