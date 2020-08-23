import numpy as np
import pandas as pd 

import re

############# MLM   MODELS
ujj_mlm1 = pd.read_csv('Input/Ujjwal/Data/with_tta.csv').rename(columns={'toxic':'mlm1'}) # Public MLM with TTA
ujj_mlm2 = pd.read_csv('Input/Ujjwal/Data/no_tta.csv').rename(columns={'toxic':'mlm2'}) # Public MLM Variant
final_mlm = pd.merge(ujj_mlm1, ujj_mlm2, on='id')
final_mlm['mlm_blend'] = (final_mlm['mlm1'] + final_mlm['mlm2'])/ 2
final_mlm = final_mlm[['id', 'mlm_blend']]
final_mlm['mlm_blend_rank'] = final_mlm['mlm_blend'].rank(pct=True)



############### MAS MODELS
mas_rank = pd.read_csv('Output/Models/Moiz/submission_MAS.csv').rename(columns={'toxic': 'mas_blend_rank'})
final_mas = mas_rank


############# MONOLINGUAL BERTS
test_sub_all = pd.read_csv('Output/Models/Moiz/submission_Roberta.csv')
df_test = pd.read_csv('Input/Igor/test.csv.zip')
test_sub_all = test_sub_all.merge(df_test[['id','lang']],on='id',how='left')
test_sub_tr = pd.read_csv('Output/Models/Igor/tr/inf_tr_google_dbmdz.prob.csv')
test_sub_tr2 = pd.read_csv('Output/Models/Igor/tr/inf_tr_google_savasy.prob.csv')
test_sub_it = pd.read_csv('Output/Models/Igor/it/inf_it_yandex_xxl.prob.csv')
test_sub_es = pd.read_csv('Output/Models/Igor/es/inf_es_google_wwm.prob.csv')
test_sub_es2 = pd.read_csv('Output/Models/Igor/es/inf_es_yandex_wwm.prob.csv')
test_sub_ru = pd.read_csv('Output/Models/Igor/ru/inf_ru_google_conv.prob.csv')
test_sub_ru2 = pd.read_csv('Output/Models/Igor/ru/inf_ru_yandex_conv.prob.csv')
test_sub_fr = pd.read_csv('Output/Models/Igor/fr/inf_fr_google_camembert_large.prob.csv')
test_sub_fr2 = pd.read_csv('Output/Models/Igor/fr/inf_fr_yandex_camembert_large.prob.csv')

test_sub_tr.columns = ['id','tr_toxic']
test_sub_tr2.columns = ['id','tr_toxic2']
test_sub_it.columns = ['id','it_toxic']
test_sub_es.columns = ['id','es_toxic']
test_sub_es2.columns = ['id','es_toxic2']
test_sub_ru.columns = ['id','ru_toxic']
test_sub_ru2.columns = ['id','ru_toxic2']
test_sub_fr.columns = ['id','fr_toxic']
test_sub_fr2.columns = ['id','fr_toxic2']

test_sub_all = test_sub_all.merge(test_sub_tr[['id','tr_toxic']],on='id',how='left')
test_sub_all = test_sub_all.merge(test_sub_tr2[['id','tr_toxic2']],on='id',how='left')
test_sub_all = test_sub_all.merge(test_sub_it[['id','it_toxic']],on='id',how='left')
test_sub_all = test_sub_all.merge(test_sub_es[['id','es_toxic']],on='id',how='left')
test_sub_all = test_sub_all.merge(test_sub_es2[['id','es_toxic2']],on='id',how='left')
test_sub_all = test_sub_all.merge(test_sub_ru[['id','ru_toxic']],on='id',how='left')
test_sub_all = test_sub_all.merge(test_sub_ru2[['id','ru_toxic2']],on='id',how='left')
test_sub_all = test_sub_all.merge(test_sub_fr[['id','fr_toxic']],on='id',how='left')
test_sub_all = test_sub_all.merge(test_sub_fr2[['id','fr_toxic2']],on='id',how='left')

######## blend monolingual with single roberta
test_sub_all['pred'] = np.where(test_sub_all.lang == 'tr',0.1*test_sub_all.toxic + 0.45*test_sub_all.tr_toxic + 0.45*test_sub_all.tr_toxic2,test_sub_all.toxic)
test_sub_all['pred'] = np.where(test_sub_all.lang == 'it',0.1*test_sub_all.pred + 0.9*test_sub_all.it_toxic,test_sub_all.pred)
test_sub_all['pred'] = np.where(test_sub_all.lang == 'es',0.1*test_sub_all.pred + 0.45*test_sub_all.es_toxic + 0.45*test_sub_all.es_toxic2,test_sub_all.pred)
test_sub_all['pred'] = np.where(test_sub_all.lang == 'ru',0.1*test_sub_all.pred + 0.45*test_sub_all.ru_toxic + 0.45*test_sub_all.ru_toxic2 ,test_sub_all.pred)
test_sub_all['pred'] = np.where(test_sub_all.lang == 'fr',0.1*test_sub_all.pred + 0.45*test_sub_all.fr_toxic + 0.45*test_sub_all.fr_toxic2,test_sub_all.pred)

final_igor = test_sub_all[['id', 'lang','pred']]
final_igor['igor_blend_rank'] = final_igor['pred'].rank(pct=True)

############## single Roberta from MAS pipeline
single_roberta = pd.read_csv('Output/Models/Moiz/submission_Roberta.csv')
single_roberta = single_roberta.rename(columns={'toxic': 'single_roberta'})
single_roberta['single_roberta_rank'] = single_roberta['single_roberta'].rank(pct=True)



############ RANK BLEND
res = pd.merge(pd.merge(pd.merge(final_mlm, final_mas, on='id'), final_igor, on='id'), single_roberta, on='id')
res['blend'] = 0.08 * res['single_roberta_rank'] + 0.29* res['mlm_blend_rank'] + 0.44*res['igor_blend_rank'] + 0.19*res['mas_blend_rank']





############# POST PROCESSING Bot messages
clusters = pd.read_csv('Input/Patrick/cluster200.csv')[['id', 'cluster']]
test = pd.read_csv('Input/Igor/test.csv.zip')
test['has_ip'] = test['content'].apply(lambda x: re.search(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', x)).map(lambda x: 0 if x is None else 1)
test['has_ip'].value_counts()

test['is_bot_msg'] = test['content'].str.strip().str.startswith('Bonjour,') & (
    test['content'].str.strip().str.endswith('(bot de maintenance)') |
    test['content'].str.strip().str.endswith('Salebot (d)'))
test.groupby('lang')['is_bot_msg'].value_counts()


res = pd.merge(res, test[['id', 'has_ip', 'is_bot_msg']], on='id')
res = pd.merge(res, clusters, on='id')

res['blend_pp'] = np.where(
    (res['has_ip']==1) & 
   (res['lang'].isin(['es', 'tr'])) & 
   ((0.5 < res['blend']) & (res['blend'] < 0.9)), 1.1*res['blend'], res['blend'])
res[['blend', 'blend_pp']].corr()


res['blend_pp'] = np.where((res['is_bot_msg']==1) , 0.5*res['blend_pp'], res['blend_pp'])
res[['blend', 'blend_pp']].corr()


res['blend_pp'] = np.where((res['cluster'].isin([39, 52, 100, 14, 111, 80, 42, 196])) , 0.5*res['blend_pp'], res['blend_pp'])
res[['blend', 'blend_pp']].corr()

res = res.rename(columns={'blend_pp':'toxic'})

res[['id', 'toxic']].to_csv('Output/Predictions/submission.csv', index=False)
