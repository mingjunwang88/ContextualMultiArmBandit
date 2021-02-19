import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt

class ORDataset(Dataset):
    ## Define the underline data source
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])
    
def DowloandfileS3(inpt, out):
    bucketname = 'fetch-data-puddle' # replace with your bucket name
    s3 = boto3.resource('s3')
    s3.Bucket(bucketname).download_file(inpt, out)
    
def read_prefix_to_df(prefix):
    s3 = boto3.resource('s3')
    bucket = s3.Bucket('fetch-data-puddle')
    prefix_objs = bucket.objects.filter(Prefix=prefix)
    prefix_df = []
    for obj in prefix_objs:
        key = obj.key
        prefix_df.append(key)
        #body = obj.get()['Body'].read()
        #df = pd.DataFrame(body)
        #prefix_df.append(df)
    #return pd.concat(prefix_df)
    return prefix_df

class OfferRec(nn.Module):
    def __init__(self, input_size, layers=5):
        super(OfferRec, self).__init__()
        
        ##In this model, I am just using the several fc layers. the last layers is a scalar and through a sigmoid transformation.
        dim_layers = []
              
        for i in range(layers):
            dim_layers.append(input_size // 2**i )
        print ('Layer distribution: ', dim_layers)
        
        fc_blocks = [self.block(i, j) for i, j in zip(dim_layers, dim_layers[1:])]
        
        self.mc = nn.Sequential(*fc_blocks)
        
        self.linear = nn.Linear(dim_layers[-1], 1)
    
    def block(self, inpt, out):
        return nn.Sequential (
            nn.Linear(inpt, out),
            nn.ReLU()
            )
    
    def forward(self, x):
        ## x is the input tensor data
        
        out = self.mc(x)
        out = self.linear(out)
        out = torch.sigmoid(out)

        return out

###Calculate the lift of the model on test dataset
class ModelLiftSpark:
    """
    testdata is a df shoud have the following cols available:
    1: actual target value
    2: expected probability
     return: a panda DF with lift column
    """
    def __init__(self,prior=0.0016, oversamp=0.33333,c=20):
        """
        prior: population trial rate
        oversamp: actual rate in the model
        c: Groups wanted
        """
        self.prior=prior
        self.oversamp=oversamp
        self.c=c
    def calLift(self,testset, target='target'):
        vector_udf = f.udf(lambda vector: vector[1].item(), DoubleType())
        self.test=testset.withColumn('prob1',vector_udf(testset.probability)).select('prob1',target)        
        w = Window.orderBy(self.test.prob1.desc()) 
        self.test=self.test.withColumn('Rank',f.ntile(self.c).over(w))
        self.test=self.test.groupBy('Rank').agg({'*':'count', 'prob1':'avg',target: 'sum'}).toPandas()
        self.test['cnt_cum']=self.test['count(1)'].cumsum()
        self.test['trial_cum']=self.test['sum(target)'].cumsum()
        
        self.test['p1']=(self.test['trial_cum']/self.test['cnt_cum']) * self.prior/self.oversamp
        self.test['p0']=(1-self.test['trial_cum']/self.test['cnt_cum']) *(1-self.prior)/(1-self.oversamp)
        self.test['p']=self.test['p1']/(self.test['p1']+self.test['p0'])
        self.test['lift']=self.test['p']/self.test['p'].values[-1].item()
        
        #self.test['p1']=(self.test['trial_cum']/self.test['cnt_cum'])
        #self.test['lift']=self.test['p1']/self.test['p1'].values[-1].item()
        
        return self.test

#model = ModelLiftSpark(prior=0.036, oversamp=0.036)
#model = ModelLiftSpark(prior=0.0025, oversamp=0.0025)
#lift = model.calLift(pred2)
#print (lift)
#spark.createDataFrame(lift).show()
#print (lift.columns)

class ModelLift:
    """
    testdata is a df shoud have the following cols available:
    1: actual target value
    2: expected probability
     return: a panda DF with lift column
    """
    def __init__(self,prior=0.0016, oversamp=0.33333,c=50):
        """
        prior: population trial rate
        oversamp: actual rate in the model
        c: Groups wanted
        """
        self.prior=prior
        self.oversamp=prior
        self.c=c
    def calLift(self,testset, target='target', prob='prob'):
        temp = testset.copy()
        temp['rank'] = pd.qcut(temp[prob], self.c , labels=False, duplicates='drop')
        
        temp['cnt'] = 1
        temp = temp.groupby('rank').agg({'cnt':'count', prob:'mean', target:'sum'}).reset_index().sort_values(by=['rank'],ascending=False)
        print (temp.head(10))
        temp['cnt_cum']=temp['cnt'].cumsum()
        temp['trial_cum']=temp['target'].cumsum()
        
        temp['p1']=(temp['trial_cum']/temp['cnt_cum']) * self.prior/self.oversamp
        temp['p0']=(1-temp['trial_cum']/temp['cnt_cum']) *(1-self.prior)/(1-self.oversamp)
        temp['p']=temp['p1']/(temp['p1']+temp['p0'])
        temp['lift']=temp['p']/temp['p'].values[-1].item()
        
        return temp

def Plot(lift):
    plt.plot(lift['rank'], lift['lift'])
    plt.title('Cumulative Lift')
    plt.xlabel('Ranks')
    plt.ylabel('Model Lift')
    plt.show()

    plt.plot(lift['rank'], lift['p1'])
    plt.title('Cumulative Respose Rate')
    plt.xlabel('Ranks')
    plt.ylabel('Response Rate')

    plt.show()
#Plot(lift)

