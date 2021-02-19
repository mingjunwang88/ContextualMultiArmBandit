
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("OfferRecommender").enableHiveSupport().getOrCreate()
print (spark.version)

####################################################
#Last step to collect all the dfs
####################################################

cols_keep = ['user_id', 'offer_id', 'redeem_date', 'value', 'id_cat_dollars', 'id_cat_units', 'id_cat_offers', 'id_cat_tenure', 'id_cat_recency', 'id_cat_points', 
'id_total_dollars', 'id_total_units', 'id_total_offers', 'id_tenure', 'id_recency', 'life', 'total_offer_dollars', 'total_offer_units', 'total_offers', 'age', 
'velocity', 'total_cat_offer_dollars', 'total_cat_offer_units', 'total_cat_offers', 'velocity_cat', 'cat_brand_sums']

#Target File
path_train ="s3://fetch-data-puddle/spark/sandbox/mingjun/OfferRecTraing"
traing_final = spark.read.parquet(path_train)

### Read the data set from s3
path ="s3://fetch-data-puddle/spark/sandbox/mingjun/OfferSeeds"
df_seeds= spark.read.parquet(path)
#print (df_seeds.count())
#df_seeds.show(n=5)

path ="s3://fetch-data-puddle/spark/sandbox/mingjun/user_cat_brds"
user_cat_brds = spark.read.parquet(path)
#print (user_cat_brds.count())
#user_cat_brds.select('user_id','offer_id','redeem_date').show(n=5)

path ="s3://fetch-data-puddle/spark/sandbox/mingjun/df_UserOffer_catbrand_wordpca"
df_UserOffer_catbrand_wordpca = spark.read.parquet(path)
#print (df_UserOffer_catbrand_wordpca.count())
#df_UserOffer_catbrand_wordpca.show(n=5)

path ="s3://fetch-data-puddle/spark/sandbox/mingjun/df_UserOffer_stateAlco"
df_UserOffer_stateAlco = spark.read.parquet(path)
print ('df_UserOffer_stateAlco : ', df_UserOffer_stateAlco.columns, df_UserOffer_stateAlco.count())
#df_UserOffer_stateAlco.show(n=5)

path ="s3://fetch-data-puddle/spark/sandbox/mingjun/df_UserOffer_facebookgogole"
df_UserOffer_facebookgogole = spark.read.parquet(path)
print ('df_UserOffer_facebookgogole: ', df_UserOffer_facebookgogole.columns, df_UserOffer_facebookgogole.count())
#df_UserOffer_facebookgogole.show(n=5)

path ="s3://fetch-data-puddle/spark/sandbox/mingjun/df_UserOffer"
df_UserOffer = spark.read.parquet(path).select(cols_keep)

cols = ['statehot','alcohot','facebook_authhot','google_authhot','user_id', 'offer_id','redeem_date']
cols_join = ['user_id','offer_id','redeem_date']

df_model = df_UserOffer_catbrand_wordpca.join(df_seeds, ['user_id','redeem_date'], 'outer').join(user_cat_brds, ['user_id','redeem_date'],'outer')
df_model = df_model.join(traing_final.select(cols_join+['target']),cols_join,'inner').fillna(0).distinct()
n_columns = len(df_model.drop('offer_id', 'user_id','redeem_date').columns)
print ('Total rows of df_model: ', df_model.count())
print ('Total columns of df_model: ', n_columns)


path_train ="s3://fetch-data-puddle/spark/sandbox/mingjun/df_UserOffer"
cols = ['user_id','offer_id','redeem_date']
df_UserOffer = spark.read.parquet(path_train)
print (df_UserOffer.columns)

df_model_other = df_UserOffer_facebookgogole.distinct().join(df_UserOffer_stateAlco.distinct(), cols_join,'outer').join(df_UserOffer, cols_join,'outer')
print ('df_model_other count: ', df_model_other.count())
print (df_model_other.columns)


#######################################################
######################################################
##Save all them
#######################################################
splits = df_model.randomSplit([1.0, 9.0], 100)
"""
path ="s3://fetch-data-puddle/spark/sandbox/mingjun/ORtest"
splits[0].repartition(200, ['user_id','offer_id','redeem_date']).write.mode('overwrite').parquet(path)

path ="s3://fetch-data-puddle/spark/sandbox/mingjun/ORtrain"
splits[1].repartition(200, ['user_id','offer_id','redeem_date']).write.mode('overwrite').parquet(path)
"""

path ="s3://fetch-data-puddle/spark/sandbox/mingjun/ORtestCSV"
splits[0].repartition(1, ['user_id','offer_id','redeem_date']).write.mode('overwrite').csv(path,header = 'true')

path ="s3://fetch-data-puddle/spark/sandbox/mingjun/ORtrainCSV"
splits[1].repartition(1, ['user_id','offer_id','redeem_date']).write.mode('overwrite').csv(path,header = 'true')

path ="s3://fetch-data-puddle/spark/sandbox/mingjun/ORotherCSV"
df_model_other = df_model_other.drop('cat_brand_sums', 'words_pca', 'statehot', 'alcohot','facebook_authhot', 'google_authhot')
df_model_other.repartition(1, ['user_id','offer_id','redeem_date']).write.mode('overwrite').csv(path,header = 'true')