import pyspark.sql.functions as f
from pyspark.sql import functions as F

from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import VectorAssembler,OneHotEncoderEstimator, StringIndexer,Tokenizer, StopWordsRemover,CountVectorizer,PCA,NGram,Word2Vec
from pyspark.sql import SparkSession

from pyspark.sql import types as T
from pyspark.ml.linalg import SparseVector, DenseVector

from pyspark.sql.window import Window
from pyspark.sql.types import FloatType,DoubleType,StringType

import pandas as pd
import numpy as np
import re
import sys

print (sys.version)

spark = SparkSession.builder.appName("OfferRecommender").enableHiveSupport().getOrCreate()
print (spark.version)

num_wks = 26
exp_days = 2
end_dt = '"2019-09-30"'
end_dt = 'current_date'
# keeping a reference to all the interim dataframes I persisted so I can free them at the very end
persisted_dataframes = []
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)


############################
##get the active users.
############################

query_t="""
select distinct 
       user_id,
       du.age as user_age, 
       du.gender as user_gender,
       u.created_date,
       du.state,
       du.state_alcohol_restriction as alco,
       du.email_domain,
       du.facebook_auth,
       du.google_auth
from fdhcpg.transaction360 i
inner join fdhdw.dim_user du using (user_id)
inner join (select id as user_id,created_date from fetchdw_v2.p_user) u using (user_id)
where datediff({}, receipt_purchase_date) <=31
and  datediff({}, receipt_purchase_date) >=0
""".format(end_dt,end_dt)

users=spark.sql(query_t).withColumn('alco',F.col('alco').cast('integer')).dropna()
print ('Total active users: ' + str(users.count()))

#######################1: Transform the category into one hot vector ###########################
def category_one_hot(name='state', df=None):
    indexer = StringIndexer(inputCol=name, outputCol=name+'int',handleInvalid='keep')
    output = indexer.fit(df).transform(df)
    docer=OneHotEncoderEstimator(inputCols=[name+'int'], outputCols=[name+'hot'])
    output=docer.fit(output).transform(output)
    return output

users = category_one_hot('state', users)
users = category_one_hot('alco', users)
users = category_one_hot('email_domain', users)
users = category_one_hot('facebook_auth', users)
users = category_one_hot('google_auth', users)

user_sample = users.sample(0.01)

user_sample.write.mode('overwrite').parquet("s3://fetch-data-puddle/spark/sandbox/mingjun/users")
user_sample = spark.read.parquet("s3://fetch-data-puddle/spark/sandbox/mingjun/users")

user_sample.createOrReplaceTempView('users_v')
print ('Total sample active users: ' + str(users.count()))

user_sample.printSchema()

###########################################
# Obtain all the barcode information
# Mostly interested in cat and brand combination
#
###########################################
query_barcode="""
select * from 
(
(select  concat_ws(' | ', category_2, brand) as cat_brand,
         barcode_id,brand,category_2, barcode from fdhdw.dim_barcode_fetch)
    inner join
       (select cat_brand,
               rank() over(order by cat_brand) as group
         from (select distinct concat_ws(' | ', category_2, brand) as cat_brand from fdhdw.{}) 
       ) using (cat_brand)
         )
         order by group
""".format('dim_barcode_fetch')

barcode_CatBrd = spark.sql(query_barcode)

###For string type input data, it is common to encode categorical features using StringIndexer first.
indexer_cat = StringIndexer(inputCol='cat_brand', outputCol="cat_brand_int",handleInvalid='keep')
barcode_CatBrd = indexer_cat.fit(barcode_CatBrd).transform(barcode_CatBrd)
### One hot encoding
docer_cat=OneHotEncoderEstimator(inputCols=['cat_brand_int'], outputCols=['cat_brand_hot'])
barcode_CatBrd=docer_cat.fit(barcode_CatBrd).transform(barcode_CatBrd)

##### Convert to dense vector to array:
def sparse_to_array(v):
  v = DenseVector(v)
  new_array = list([float(x) for x in v])
  return new_array
sparse_to_array_udf = F.udf(sparse_to_array, T.ArrayType(T.FloatType()))
barcode_CatBrd = barcode_CatBrd.withColumn('cat_brand_array', sparse_to_array_udf('cat_brand_hot'))
barcode_CatBrd.show(n=5)

path_train ="s3://fetch-data-puddle/spark/sandbox/mingjun/barcode_CatBrd"
## Save the file for later use
barcode_CatBrd.write.parquet(path_train,mode="overwrite")


barcode_CatBrd=spark.read.parquet(path_train).select(['barcode_id','barcode', 'cat_brand','group','cat_brand_array'])

barcode_CatBrd.show(n=5)
n_brds = barcode_CatBrd.select('group').distinct().count()
print ('Total Ca_Brds: ',n_brds)
barcode_CatBrd.createOrReplaceTempView('barcode_CatBrd_v')

############################
##get all the offer information including triggered, segmented offers and national offers and other types that active in the current date
## Mostly intersted in the count of the cat_brand
## 
############################

query="""
SELECT      promo_id AS offer_id
		,	elgreq_start_date AS start_date
		,	elgreq_end_date AS end_date
		,	category AS offer_category
		,	points_earned / 1000.0	as value
		,   date_sub(current_date, 13*7) as offer_cutoff
		,   plan_tactic as tactic
        ,	plan_audience
        ,   elgreq_segment
        ,   payer_name
        ,   details
        ,   cat_brand
        ,   cat_brand_array
   from fdhdw.dim_promo a 
   inner join barcode_CatBrd_v b using (barcode)
where datediff(current_date, elgreq_start_date) between 30 and 6*4*7
and datediff(date_sub(current_date, 0*7),elgreq_end_date) > 0
""".format()


query2="""
SELECT id AS offer_id
		,	eligibility_start_date AS start_date
		,	eligibility_end_date AS end_date
		,	display_fr_category AS offer_category
		,	benefit_points_earned / 1000.0	as value
		,   date_sub(current_date, 13*7) as offer_cutoff
		,   1 as train
		,   planning_tactic as tactic,
		planning_offer_audience as plan_audience,
		eligibility_segment as elgreq_segment
   from fetchdw_v2.p_offer
where display_fr_description is not null
and datediff(current_date, eligibility_start_date) between 30 and 6*4*7
and datediff(date_sub(current_date, 0*7),eligibility_end_date) > 0
"""

#and planning_offer_audience in ('NATIONAL', 'null')


df_offers = spark.sql(query)
df_offers = df_offers.groupby(['offer_id','start_date', 'end_date', 'offer_category','value','offer_cutoff','tactic', 'plan_audience','elgreq_segment','payer_name','details'])

### Create the total the count of eacg cat/brand combination for each offers.
df_offers = df_offers.agg(F.array(*[F.sum(F.col('cat_brand_array')[i]) for i in range(n_brds)]).alias("cat_brand_sums"))

###############################3: Transform the short term description in to vector ###########
## Remove the Space using UDF fun
def remov(x):
    a=x.replace('OZ','').replace('CT','').replace('.','').replace('%','').replace('(','').replace(')','').replace('-','').replace('$','')
    a=re.sub('[0-9]','',a)
    a=re.sub('&','',a)
    return a
removD=F.udf(remov,StringType() )       

##String cleaning first
df_offers=df_offers.withColumn('details',removD(F.col('details')))

"""
## Try to use pipeline 
#tokenizer = Tokenizer(inputCol="descr", outputCol="words")
#stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")
#countVectors = CountVectorizer(inputCol="words", outputCol="features", vocabSize=1000, minDF=5)
#pca=PCA(k=100, inputCol="features", outputCol="pcaFeatures")
#pipeline=Pipeline(stages=[tokenizer,stopwordsRemover,countVectors,pca])
#barcode=pipeline.fit(barcode).transform(barcode)
"""

##Tokenize
tokenizer = Tokenizer(inputCol="details", outputCol="words")
df_offers=tokenizer.transform(df_offers)
#barcode.show(n=5)
#barcode.printSchema()

# stop words removal
stopwordsRemover = StopWordsRemover(inputCol="words", outputCol="filtered")
df_offers=stopwordsRemover.transform(df_offers).repartition(1000, 'offer_id')
#barcode.drop('barcode').show(n=5)


#######################
# Try not to use BOW and wor2Vector from Spark since this does not fit with the model objective. Instead need to build the Imbedding tables that need to be estimated in the larger archticture
######################

# bag of words count
countVectors = CountVectorizer(inputCol="filtered", outputCol="word_count",vocabSize=10000, minDF=5).fit(df_offers)
df_offers=countVectors.transform(df_offers)
print ('Total columns: ', len(df_offers.columns))
#df_offers.show()
lenth=len(countVectors.vocabulary)
print ('Toal Vcob size: ', lenth)

## Aply the PCA transformation
pca=PCA(k=100, inputCol="word_count", outputCol="words_pca").fit(df_offers)
df_offers = pca.transform(df_offers)

"""
## apply word2Vector
word2Vec = Word2Vec(vectorSize=128, minCount=0, inputCol="filtered", outputCol="descr_embed")
model = word2Vec.fit(barcode)
barcode = model.transform(barcode)
"""

df_offers = df_offers.drop('details','words','filtered','word_count').sample(fraction = 0.4)
df_offers.write.mode("overwrite").parquet("s3://fetch-data-puddle/spark/sandbox/mingjun/df_offers")
print ('{} was writen to s3'.format('df_offers'))


df_offers = spark.read.parquet('s3://fetch-data-puddle/spark/sandbox/mingjun/df_offers')
df_offers.groupby('plan_audience').count().show(n=10)
print ('Total number offers: ', df_offers.count())
df_offers.where(f.col('plan_audience').isin('NATIONAL','SEGMENTED')).groupBy('offer_category').count().show(n=10, truncate=False)

## Note: Only need barcode,category_hot,manuf_hot etc.

df_offers.printSchema()
df_offers.createOrReplaceTempView('df_offers_v')

df_offers.show(n=5)

#############################
#Obtain all the IDs
#############################
## Take the user and ID combinatipon. new User and PURCHASED_BASED? 
#wh = 'planning_offer_audience not in ("SEGMENTED")'
wh = 'plan_audience in ("NATIONAL")'
query="""
    select user_id,
       offer_id,
       start_date,
       end_date,
       offer_category,
       0 as offer_seg,
       value,
       tactic,
       plan_audience,
       user_age, 
       user_gender
    from users_v
    cross join (select * from df_offers_v where plan_audience in ("NATIONAL"))
    where datediff(start_date, created_date) > 0 
    """.format(wh)
#    where datediff(start_date, created_date) > 0    
user_offer_nonseg=spark.sql(query)
#print ('Total random offer and ID combination: ',user_offer_nonseg.count() )
user_offer_nonseg.repartition(1000, ['user_id','offer_id']).write.mode("overwrite").parquet("s3://fetch-data-puddle/spark/sandbox/mingjun/user_offer_nonseg")
#user_offer_nonseg.write.mode("overwrite").parquet("hdfs:///useroffernonseg")
print ('user_offer_nonseg saved')
#spark.read.parquet("s3://fetch-data-puddle/spark/sandbox/brian/active_users_sample_v").createOrReplaceTempView('active_users_sample_v')

### Take all the Segmented offer and IDs information
query = """
select  a.user_id,
        o.offer_id,
		o.start_date,
		o.end_date ,
        o.offer_category,
        1 as offer_seg,
        o.value,
        o.tactic,
        o.plan_audience,
        du.age as user_age, 
        du.gender as user_gender
from fetchdw_v2.p_user_segment a
inner join df_offers_v o on (a.segment_name =o.elgreq_segment)
inner join users_v using (user_id)
inner join fdhdw.dim_user du using (user_id)
WHERE o.plan_audience in ("SEGMENTED")
""".format()

user_offer_seg = spark.sql(query)
#print ('Total Segmented IDs: ',user_offer_seg.count() )
user_offer_seg.repartition(1000, ['user_id','offer_id']).write.mode("overwrite").parquet("s3://fetch-data-puddle/spark/sandbox/mingjun/user_offer_seg")
#user_offer_seg.write.mode("overwrite").parquet("hdfs:///userofferseg")    #hdfs:///OfferRecommendationData
print ('user_offer_seg saved')


### Take all the triggered offer and IDs information
#p_user_trigger=spark.read.parquet("s3://fetch-analytics-hub/data/prod/an/pysana_ueo_ueo").select('user_id','offer_id').distinct()
#p_user_trigger.createOrReplaceTempView('p_user_trigger_v')
#p_user_trigger.printSchema()

query = """
select  a.user_id,
        o.offer_id,
		o.start_date,
		o.end_date ,
        o.offer_category,
        2 as offer_seg,
        o.value,
        o.tactic,
        o.plan_audience,
        du.age as user_age, 
        du.gender as user_gender
from p_user_trigger_v a
inner join df_offers_v o on (a.offer_id =o.offer_id)
inner join users_v using (user_id)
inner join fdhdw.dim_user du using (user_id)
WHERE upper(o.plan_audience) in ("PURCHASED_BASED")
""".format(end_dt)

#user_offer_trigger = spark.sql(query).sample(0.001)
#print ('Total triger offers IDs: ',user_offer_trigger.count() )

user_offer_seg = spark.read.parquet("s3://fetch-data-puddle/spark/sandbox/mingjun/user_offer_seg")
user_offer_nonseg = spark.read.parquet("s3://fetch-data-puddle/spark/sandbox/mingjun/user_offer_nonseg")

#user_offer_seg = spark.read.parquet("hdfs://userofferseg")
#user_offer_nonseg = spark.read.parquet("hdfs://useroffernonseg")
#"hdfs://user_offer_seg" 

user_offer= user_offer_nonseg.union(user_offer_seg)

user_offer.write.mode('overwrite').parquet("s3://fetch-data-puddle/spark/sandbox/mingjun/user_offer")
user_offer = spark.read.parquet("s3://fetch-data-puddle/spark/sandbox/mingjun/user_offer")

user_offer.printSchema()
user_offer.createOrReplaceTempView('user_offer_v')

print ('Total rows of final data: ', user_offer.count())
user_offer.show(n=5, truncate= False)

print ('Average number of offer each person:')
user_offer.groupby('user_id').count().agg({'count':'avg'}).show()


################################
#Obtain the target 
################################
query_targ ="""
select user_id,
       offer_id,
       offer_category,

       case 
       when award_date is null then 0
       else 1 
       end 
       as target,
       
       nvl(award_date,end_date ) as redeem_date_old,
       
       award_date,
       start_date,
       end_date,
       offer_seg,
       value,
       tactic,
       user_age, 
       user_gender
   from 
	   (       
       	select *
	      From user_offer_v uo 
	      left join (select promo_id as offer_id, user_id,award_date from fdhcpg.promotion180) using (offer_id, user_id)
    	) a
"""

## Note: redeem_date is to be used for last day to collect information
df_targ=spark.sql(query_targ).drop('award_date').withColumn('redeem_date', f.to_date(f.col('redeem_date_old'))).drop('redeem_date_old').persist()
persisted_dataframes.append(df_targ)

df_targ.createOrReplaceTempView('df_targ_v')
prior = df_targ.agg({'target': 'avg'}).toPandas().values[0][0]
print ('The target prior is: ', prior)
#print ('The target prior is: ', df_targ.agg({'target': 'avg'}).collect()[0])
#df_targ.where(f.col('train') == 0).groupby('user_id').count().show(n=5)
print(df_targ.columns)
print ('Total Orginal target file: ', df_targ.count())
df_targ.show(n=5)


###################################
# Oversample
###################################
#Resample the training data
all_ones = df_targ.where('target ==1')
num_ones = all_ones.count()
print (num_ones)

all_zeros = df_targ.where('target ==0')
num_zeros = all_zeros.count()
print (num_zeros)

## Everyone one has 10 zeros
frac =1.0 * num_ones / num_zeros *10
traing_final = all_zeros.sample(fraction=frac, seed=100).union(all_ones).fillna(0)

traing_final.groupby('target').count().show()
traing_final.agg({'target':'avg'}).show()

path_train ="s3://fetch-data-puddle/spark/sandbox/mingjun/OfferRecTraing"
## Save the file for later use
traing_final.write.parquet(path_train,mode="overwrite")

## Save the file for later use
traing_final = spark.read.parquet(path_train)
traing_final.createOrReplaceTempView('traing_final_v')

traing_final.show(n=5)
traing_final.count()


############################
# Obtain the seed information
############################
query ="""
select user_id,
        offer_id,
        redeem_date,
        nvl(tenure_seeds,0) as tenure_seeds,
        nvl(recency_seeds,0) as recency_seeds,
        nvl(dollars_seeds,0) as dollars_seeds,
        nvl(units_seeds,0) as units_seeds,
        nvl(trips_seeds,0) as trips_seeds
from traing_final_v left join 
(
select  user_id,
        offer_id,
        redeem_date,
        nvl(max(datediff(redeem_date,receipt_purchase_date)),0) as tenure_seeds,
        nvl(max(datediff(redeem_date,receipt_purchase_date)),0) as recency_seeds,
        nvl(sum(item_discounted_unit_price * item_quantity),0) as dollars_seeds,
        nvl(sum(item_quantity),0) as units_seeds,
        nvl(count(distinct receipt_id),0) as trips_seeds
    from (select * from 
    (select user_id, offer_id, redeem_date, barcode as barcode_upc from traing_final_v u inner join fdhdw.dim_promo o on (u.offer_id = o.promo_id) ) a
    left join (select * from fdhcpg.transaction360) t using (user_id, barcode_upc)
    where datediff(redeem_date, receipt_purchase_date) <={}*7
    )
    group by user_id, offer_id, redeem_date
    ) a using ( user_id,offer_id, redeem_date)
""".format(num_wks)
df_seeds = spark.sql(query)

path_train ="s3://fetch-data-puddle/spark/sandbox/mingjun/OfferSeeds"
df_seeds.write.parquet(path_train,mode="overwrite")
df_seeds = spark.read.parquet(path_train)

print (df_seeds.count())
df_seeds.show(n=5)


###############################
# Obtain all the cat/brand combination information for each user. 
# This is use matching with offer cat_brand_sum
################################

query = """
select user_id, 
       redeem_date
"""
for i in range(1, n_brds+1):
    query+= """, sum(case when group == {} then item_quantity*item_discounted_unit_price end) as id_catBrd_{}""".format(i, i)

query = query + """
from  traing_final_v  u 
      inner join fdhcpg.transaction360 t using (user_id)
      inner join barcode_CatBrd_v using (barcode_id)
    where datediff(redeem_date, receipt_scan_date) between 1 and 7*{}
    group by 1,2
""".format(num_wks)

user_cat_brds = spark.sql(query)
user_cat_brds.write.mode('overwrite').parquet('s3://fetch-data-puddle/spark/sandbox/mingjun/user_cat_brds')
print ('user_cat_brds saved')

###############################
# User level infoamrion
###############################
### Used for colleting category user category information
query_targG="""
select distinct user_id,
       offer_category,
       redeem_date
  from traing_final_v
"""
df_targG=spark.sql(query_targG)
df_targG.createOrReplaceTempView('df_targG_v')

## Used for collecting overall information
query_targA="""
select distinct user_id,
       redeem_date
  from traing_final_v
"""
df_targA=spark.sql(query_targA)
df_targA.createOrReplaceTempView('df_targA_v')


####################################
##User lever information
###################################

## Overall level information  Match with target file with user_id, redeem_date, category
query_overal ="""
select t.user_id,
       redeem_date,
       sum(receipt_item_unit_price * receipt_item_quantity) as id_total_dollars,
       sum(receipt_item_quantity) as id_total_units,
       count(promo_id) as id_total_offers,
       max(datediff(redeem_date, award_date)) as id_tenure,
       min(datediff(redeem_date, award_date)) as id_recency
   from df_targA_v t inner join
	   (select a.user_id, a.promo_id, actreq_dollar_amount,actreq_quantity,award_date,promo_category as offer_category, receipt_item_unit_price, receipt_item_quantity
	      From fdhcpg.promotion180 a
	      where  award_date is not null
    	)  a using (user_id )
        where datediff(redeem_date, award_date) <={}*7
        and datediff(redeem_date, award_date) > 0
        group by 1,2
""".format(num_wks)

query_cat ="""
select t.user_id,
       t.offer_category,
       redeem_date,
       sum(receipt_item_unit_price * receipt_item_quantity) as id_cat_dollars,
       sum(receipt_item_quantity) as id_cat_units,
       count(promo_id) as id_cat_offers,
       max(datediff(redeem_date, award_date)) as id_cat_tenure,
       min(datediff(redeem_date, award_date)) as id_cat_recency,
       sum(award_points) as id_cat_points
   from df_targG_v t inner join
	   (select a.user_id, a.promo_id, actreq_dollar_amount,actreq_quantity,award_date,promo_category as offer_category,receipt_item_unit_price, receipt_item_quantity,award_points
	      From fdhcpg.promotion180 a
	      where  award_date is not null
    	)  a using (user_id, offer_category )
        where datediff(redeem_date, award_date) <={}*7
        and datediff(redeem_date, award_date) > 0
        group by 1,2,3
""".format(num_wks)

## Category points information
query_points ="""
select t.user_id,
       t.offer_category,
       redeem_date,
       sum(award_points) as id_cat_points
   from df_targG_v t inner join
	   (select a.user_id, a.promo_id, actreq_dollar_amount,actreq_quantity,award_date,promo_category as offer_category,award_points
	      From fdhcpg.promotion180 a
	      where  award_date is not null
    	)  a using (user_id, offer_category )
        where datediff(redeem_date, award_date) <={}*7
        and datediff(redeem_date, award_date) > 0
        group by 1,2,3
""".format(num_wks)

df_overal=spark.sql(query_overal)

print ('Total overal: ', df_overal.count())

df_cat=spark.sql(query_cat)

df_user=traing_final.join(df_cat,['user_id','redeem_date','offer_category'],'left').join(df_overal,['user_id','redeem_date'],'left').fillna(0)
df_user.write.mode('overwrite').parquet("s3://fetch-data-puddle/spark/sandbox/mingjun/df_user")

df_user = spark.read.parquet("s3://fetch-data-puddle/spark/sandbox/mingjun/df_user")
print (df_user.columns)
print ('Total number of user side obs:', df_user.count() )


####################################
##Offer lever information
###################################
traing_final.createOrReplaceTempView('traing_final_v')

## Used for collecting offer level information
query_offer="""
select distinct offer_id,
       start_date,
       end_date,
       redeem_date
from traing_final_v
"""
df_offer=spark.sql(query_offer)
df_offer.createOrReplaceTempView('df_offer_v')
print ('df_offer: ', df_offer.count())

## Use for collect category offer information
query_offer_cat="""
select distinct offer_category,
       redeem_date
from traing_final_v
"""
df_offer_cat=spark.sql(query_offer_cat)
df_offer_cat.createOrReplaceTempView('df_offer_cat_v')
print ('df_offer_cat: ', df_offer_cat.count())
df_offer_cat.withColumn('redeem_date2', f.to_date(f.col('redeem_date'))).orderBy('redeem_date2').show(truncate=False, n=5)


##################### Overall information match with offer_id, redeem_date
query_overal ="""
select offer_id,
       redeem_date,
       start_date,
       end_date,
       datediff(end_date, start_date) as life,
       sum(receipt_item_unit_price * receipt_item_quantity) as total_offer_dollars,
       sum(receipt_item_quantity) as total_offer_units,
       count(user_id) as total_offers,
       datediff(redeem_date,start_date)+1 as age
   from df_offer_v t inner join
	   ( select a.user_id, a.promo_id as offer_id, actreq_dollar_amount,actreq_quantity,award_date,promo_category as offer_category,award_points,receipt_item_unit_price, receipt_item_quantity
	      From fdhcpg.promotion180 a
	      where  award_date is not null
    	) a using (offer_id)
    where datediff(redeem_date, award_date) <={}*7
	  and datediff(redeem_date, award_date) > 0
    group by 1,2,3,4,5
""".format(num_wks)

df_offer_overal=spark.sql(query_overal).drop('start_date','end_date')

df_offer_overal.write.parquet("s3://fetch-data-puddle/spark/sandbox/mingjun/df_offer_overal",mode="overwrite")

df_offer_overal = spark.read.parquet("s3://fetch-data-puddle/spark/sandbox/mingjun/df_offer_overal")
df_offer_overal=df_offer_overal.withColumn('velocity', df_offer_overal.total_offers/df_offer_overal.age)


##################### Category information match with offer_category, redeem_date
query_cat ="""
select offer_category,
       redeem_date,
       sum(receipt_item_unit_price * receipt_item_quantity) as total_cat_offer_dollars,
       sum(receipt_item_quantity) as total_cat_offer_units,
       count(offer_id) as total_cat_offers
   from df_offer_cat_v t inner join
	   (select a.user_id, a.promo_id as offer_id, actreq_dollar_amount,actreq_quantity,award_date,promo_category as offer_category,award_points,receipt_item_unit_price, receipt_item_quantity
	      From fdhcpg.promotion180 a
	      where  award_date is not null
    	) a using (offer_category)
    where datediff(redeem_date, award_date) <={}*7
	  and datediff(redeem_date, award_date) > 0
    group by 1,2
""".format(num_wks)

df_offer_cat=spark.sql(query_cat)
df_offer_cat.write.parquet("s3://fetch-data-puddle/spark/sandbox/mingjun/df_offer_cat",mode="overwrite")
print ('df_offer_cat saved')
df_offer_cat = spark.read.parquet("s3://fetch-data-puddle/spark/sandbox/mingjun/df_offer_cat")
df_offer_cat=df_offer_cat.withColumn('velocity_cat', df_offer_cat.total_cat_offers/(26*7))

##############################
#Start to Write out
##############################

df_UserOffer = df_user.join(df_offer_overal ,['offer_id','redeem_date'],'left').join(df_offer_cat, ['offer_category','redeem_date'], 'inner').fillna(0)
path_train ="s3://fetch-data-puddle/spark/sandbox/mingjun/df_UserOffer"
df_UserOffer =df_UserOffer.join(df_offers.drop('value','alcoint','stateint','alco','state','created_date','user_gender','user_age','email_domainhot'), ['offer_id'], 'inner').join(users.drop('elgreq_segment','plan_audience','offer_cutoff','offer_category','end_date','start_date'), ['user_id'], 'inner')
df_UserOffer = df_UserOffer.drop('elgreq_segment','plan_audience','offer_cutoff','offer_category','end_date','start_date','email_domain','payer_name','tactic','email_domainhot')
df_UserOffer = df_UserOffer.drop('alcoint','stateint','alco','state','created_date','user_gender','user_age','google_authint','facebook_authint','email_domainint','facebook_auth','google_auth')

df_UserOffer.write.parquet(path_train,mode="overwrite")
print ('df_UserOffer saved')


###############################
# Join more information
###############################
def sparse_to_array(v):
  v = DenseVector(v)
  new_array = list([float(x) for x in v])
  return new_array
sparse_to_array_udf = F.udf(sparse_to_array, T.ArrayType(T.FloatType()))

def decomp(in_put, cols, col_type='temp'):
    ## Decompose a array column into multiple colummns
    
    if col_type =='array':
        output = in_put.select(cols)
        lenth = len(output.take(1)[0][0])
        output = in_put.select([in_put['user_id'],in_put['offer_id'],in_put['redeem_date']]+[in_put[cols][i] for i in range(lenth)])
    else:  ## When the column is a dense vector
        in_put = in_put.withColumn(cols+'_', sparse_to_array_udf(cols))
        
        output = in_put.select(cols+'_')
        lenth = len(output.take(1)[0][0])
        
        output = in_put.select([in_put['user_id'],in_put['offer_id'],in_put['redeem_date']] + [in_put[cols+'_'][i] for i in range(lenth)])
        
    return output


#################################
# Unpack all the array and vect columns
#################################
df_UserOffer = spark.read.parquet(path_train).select('user_id','offer_id','redeem_date','cat_brand_sums','words_pca')

## This data include cat_brand_sums and words_pca
path = "s3://fetch-data-puddle/spark/sandbox/mingjun/df_UserOffer_catbrand_wordpca"
df_UserOffer =df_UserOffer.join(decomp(df_UserOffer, 'cat_brand_sums', 'array'), ['user_id','offer_id','redeem_date'], 'inner').join(decomp(df_UserOffer, 'words_pca'), ['user_id','offer_id','redeem_date'], 'inner')
df_UserOffer.drop('cat_brand_sums','words_pca').repartition(1000,['user_id','offer_id','redeem_date']).write.parquet(path,mode="overwrite")
print ('df_UserOffer_catbrand_wordpca saved' )


## Now collect all the others
"""
cols = ['statehot','alcohot','facebook_authhot','google_authhot','user_id', 'offer_id','redeem_date']
cols_join = ['user_id','offer_id','redeem_date']
df_UserOffer = spark.read.parquet(path_train).select(cols)

df_UserOffer =df_UserOffer.join(decomp(df_UserOffer, 'statehot'), cols_join, 'inner').join(decomp(df_UserOffer, 'alcohot'), cols_join, 'inner')
df_UserOffer =df_UserOffer.join(decomp(df_UserOffer, 'facebook_authhot'), cols_join, 'inner').join(decomp(df_UserOffer, 'google_authhot'), cols_join, 'inner')
df_UserOffer = df_UserOffer.drop('statehot','alcohot','facebook_authhot','google_authhot')
# Save it
print ('Total final columns: ', len(df_UserOffer.columns))
path = "s3://fetch-data-puddle/spark/sandbox/mingjun/df_UserOffer_others"
df_UserOffer.repartition(1000, cols_join).write.parquet(path,mode="overwrite")
print ('df_UserOffer_others saved')
"""


