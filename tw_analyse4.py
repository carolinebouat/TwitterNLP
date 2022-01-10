########## This code was used to perform two different statistical tests that were core to our analysis
########## First, we transformed our collected data into a dataframe that we cleaned to get the proportion of vaccine-related tweets 
########## Then, we tested our hypothesis 1: compare the frequency of vaccine-related tweets between our different locations
########## Finally, hypothesis 2 was explored: examine the sentiment score of these vaccine-related tweets according to location
########## Our results are presented in the form of graphs which are also displayed in our report


# import all necessary modules/functions

import pickle
import pandas as pd
import numpy as np
import re
from statsmodels.stats.proportion import proportion_confint
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import proportion_effectsize
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob 
from textblob_fr import PatternTagger, PatternAnalyzer 
from scipy.stats import normaltest
from scipy.stats import ranksums
import pingouin as pg
from scipy.stats import chi2_contingency



# all our data collection was combined in three different pickle files according to the different locations 

# read tweets from 3 pickle files into 3 lists

file_britany = 'britany_all_combined.pickle'
file_bouches = 'bouches_all_combined.pickle'
file_france = 'france_all_combined.pickle'

tweets_britany_list = pickle.load(open(file_britany, 'rb'))
tweets_bouches_list = pickle.load(open(file_bouches, 'rb'))
tweets_france_list = pickle.load(open(file_france, 'rb'))



# convert lists to pd dataframes, and get the total number of tweets collected for each location

print('\n\nRaw data - Number of tweets:')

tweets_britany_df = pd.DataFrame(tweets_britany_list)
number_britany = len(tweets_britany_df)

tweets_bouches_df = pd.DataFrame(tweets_bouches_list)
number_bouches = len(tweets_bouches_df)

tweets_france_df = pd.DataFrame(tweets_france_list)
number_france = len(tweets_france_df)

print('\nBritany: ' + str(number_britany) + ', Bouches-du-Rhône: ' + str(number_bouches) + ', France: ' + str(number_france))



# create new column in each dataframe for full texts of tweets, remove duplicates tweets 
# gives us the updated total number of tweets collected for each location

raw_text_britany = []

for i in range(number_britany):
	if tweets_britany_df.iloc[i]['truncated'] == True:
		raw_text_britany.append(tweets_britany_df.iloc[i]['extended_tweet']['full_text'])
	else:
		raw_text_britany.append(tweets_britany_df.iloc[i]['text'])

tweets_britany_df['raw_text'] = raw_text_britany

tweets_britany_df = tweets_britany_df.drop_duplicates(subset=['id_str']) 
number_britany = len(tweets_britany_df)

raw_text_bouches = []

for i in range(number_bouches):
	if tweets_bouches_df.iloc[i]['truncated'] == True:
		raw_text_bouches.append(tweets_bouches_df.iloc[i]['extended_tweet']['full_text'])
	else:
		raw_text_bouches.append(tweets_bouches_df.iloc[i]['text'])

tweets_bouches_df['raw_text'] = raw_text_bouches

tweets_bouches_df = tweets_bouches_df.drop_duplicates(subset=['id_str'])
number_bouches = len(tweets_bouches_df)

raw_text_france = []

for i in range(number_france):
	if tweets_france_df.iloc[i]['truncated'] == True:
		raw_text_france.append(tweets_france_df.iloc[i]['extended_tweet']['full_text'])
	else:
		raw_text_france.append(tweets_france_df.iloc[i]['text'])

tweets_france_df['raw_text'] = raw_text_france

tweets_france_df = tweets_france_df.drop_duplicates(subset=['id_str'])
number_france = len(tweets_france_df)



# define keywords and function to check if tweets contain them
# any word that contains one of the keywords will be kept, for example, a tweet containing "antivax" will give True

keywords = ['vaccin', 'vax']

def contains(original, targets):
	for target in targets:
		if target in original:
			return True



# create new column in each dataframe for whether tweets contain keyword(s)

vaccin_britany = []

for tweet in tweets_britany_df['raw_text']:
	if contains(tweet, keywords) == True:
		vaccin_britany.append(True)
	else:
		vaccin_britany.append(False)

tweets_britany_df['vaccin'] = vaccin_britany

vaccin_bouches = []

for tweet in tweets_bouches_df['raw_text']:
	if contains(tweet, keywords) == True:
		vaccin_bouches.append(True)
	else:
		vaccin_bouches.append(False)

tweets_bouches_df['vaccin'] = vaccin_bouches

vaccin_france = []

for tweet in tweets_france_df['raw_text']:
	if contains(tweet, keywords) == True:
		vaccin_france.append(True)
	else:
		vaccin_france.append(False)

tweets_france_df['vaccin'] = vaccin_france



# for each region, print number of tweets that contain keyword(s), add normalize=True to get the proportions

print('\n\nRQ 1 - Frequency of vaccine-related tweets:')

prop_britany = tweets_britany_df['vaccin'].value_counts()[1]/number_britany * 100
ci_britany = proportion_confint(count=tweets_britany_df['vaccin'].value_counts()[1], nobs=number_britany, alpha=0.05)
print('\nBritany:')
print(str(tweets_britany_df['vaccin'].value_counts()[1]) + ' out of ' + str(number_britany) + ' tweets (' + str(prop_britany) + '%)')
print('95CI is ' + str(ci_britany[0]*100) + '%, ' + str(ci_britany[1]*100) + '%')

prop_bouches = tweets_bouches_df['vaccin'].value_counts()[1]/number_bouches * 100
ci_bouches = proportion_confint(count=tweets_bouches_df['vaccin'].value_counts()[1], nobs=number_bouches, alpha=0.05)
print('\nBouches-du-Rhône:')
print(str(tweets_bouches_df['vaccin'].value_counts()[1]) + ' out of ' + str(number_bouches) + ' tweets (' + str(prop_bouches) + '%)')
print('95CI is ' + str(ci_bouches[0]*100) + '%, ' + str(ci_bouches[1]*100) + '%')

prop_france = tweets_france_df['vaccin'].value_counts()[1]/number_france * 100
ci_france = proportion_confint(count=tweets_france_df['vaccin'].value_counts()[1], nobs=number_france, alpha=0.05)
print('\nFrance (control):')
print(str(tweets_france_df['vaccin'].value_counts()[1]) + ' out of ' + str(number_france) + ' tweets (' + str(prop_france) + '%)')
print('95CI is ' + str(ci_france[0]*100) + '%, ' + str(ci_france[1]*100) + '%')



# print statistical tests for frequency (hypothesis 1)

counts = np.array([tweets_britany_df['vaccin'].value_counts()[1], tweets_bouches_df['vaccin'].value_counts()[1]])
nobs = np.array([number_britany, number_bouches])
stat, pval = proportions_ztest(counts, nobs)
effsize = abs(proportion_effectsize(tweets_britany_df['vaccin'].value_counts()[1]/number_britany, tweets_bouches_df['vaccin'].value_counts()[1]/number_bouches))

print('\nComparing Britany & Bouches-du-Rhône:')
print('stat=' + str(abs(stat)) + ', pval=' + str(pval) + ', h=' + str(effsize))

counts = np.array([tweets_britany_df['vaccin'].value_counts()[1], tweets_france_df['vaccin'].value_counts()[1]])
nobs = np.array([number_britany, number_france])
stat, pval = proportions_ztest(counts, nobs)
effsize = abs(proportion_effectsize(tweets_britany_df['vaccin'].value_counts()[1]/number_britany, tweets_france_df['vaccin'].value_counts()[1]/number_france))

print('\nComparing Britany & France (control):')
print('stat=' + str(abs(stat)) + ', pval=' + str(pval) + ', h=' + str(effsize))

counts = np.array([tweets_bouches_df['vaccin'].value_counts()[1], tweets_france_df['vaccin'].value_counts()[1]])
nobs = np.array([number_bouches, number_france])
stat, pval = proportions_ztest(counts, nobs)
effsize = abs(proportion_effectsize(tweets_bouches_df['vaccin'].value_counts()[1]/number_bouches, tweets_france_df['vaccin'].value_counts()[1]/number_france))

print('\nComparing Bouches-du-Rhône & France (control):')
print('stat=' + str(abs(stat)) + ', pval=' + str(pval) + ', h=' + str(effsize))



# show graphs for frequency (hypothesis 1)

prop_all = [['Britany', prop_britany], ['Bouches-du-Rhône', prop_bouches], ['France (control)', prop_france]]
prop_all_df = pd.DataFrame(prop_all, columns = ['Region','Vaccine-related Tweets (in %)'])
prop_tweet = ['224 / 15 025', '507 / 15 028', '328 / 15 119']
error_ci_all = [(ci_britany[1]-ci_britany[0])*100/2, (ci_bouches[1]-ci_bouches[0])*100/2, (ci_france[1]-ci_france[0])*100/2]

prop_all_df.plot.bar(x = 'Region', rot = 0, y = None, legend = None, yerr = error_ci_all, capsize = 10, color = 'darkcyan')
plt.title('Proportion of vaccine-related tweets, by region', fontsize = 18, fontweight = 'bold')
plt.xlabel("Region", fontsize = 12, fontweight = 'bold')
plt.ylabel("Vaccine-related Tweets (in %)", fontsize = 12, fontweight = 'bold')
plt.text(0, prop_all_df['Vaccine-related Tweets (in %)'][0]-0.75, prop_tweet[0], ha = 'center', bbox = dict(facecolor = 'white', alpha =.8))
plt.text(1, prop_all_df['Vaccine-related Tweets (in %)'][1]-0.75, prop_tweet[1], ha = 'center', bbox = dict(facecolor = 'white', alpha =.8))
plt.text(2, prop_all_df['Vaccine-related Tweets (in %)'][2]-0.75, prop_tweet[2], ha = 'center', bbox = dict(facecolor = 'white', alpha =.8))
plt.show()





# create new dataframes with only vaccine-related tweets collected

tweets_vaccin_britany = tweets_britany_df[tweets_britany_df['vaccin']==True]
tweets_vaccin_bouches = tweets_bouches_df[tweets_bouches_df['vaccin']==True]
tweets_vaccin_france = tweets_france_df[tweets_france_df['vaccin']==True]



# clean tweet text before sentiment analysis 

def clean_text(text):
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', '')
    text = ' '.join(text.split())
    text = re.sub(r"[A-Za-z\.]*[0-9]+[A-Za-z%°\.]*", "", text)
    text = re.sub(r"(\s\-\s|-$)", "", text)
    text = re.sub(r"[,\!\?\%\(\)\/\"]", "", text)
    text = re.sub(r"\&\S*\s", "", text)
    text = re.sub(r"\&", "", text)
    text = re.sub(r"\+", "", text)
    text = re.sub(r"\#", "", text)
    text = re.sub(r"\$", "", text)
    text = re.sub(r"\£", "", text)
    text = re.sub(r"\%", "", text)
    text = re.sub(r"\:", "", text)
    text = re.sub(r"\@", "", text)
    text = re.sub(r"\-", "", text)
    return text

clean_text_vaccin_britany = []

for text in tweets_vaccin_britany['raw_text']:
	cleaned_text = clean_text(text)
	clean_text_vaccin_britany.append(cleaned_text)

clean_text_vaccin_bouches = []

for text in tweets_vaccin_bouches['raw_text']:
	cleaned_text = clean_text(text)
	clean_text_vaccin_bouches.append(cleaned_text)

clean_text_vaccin_france = []

for text in tweets_vaccin_france['raw_text']:
	cleaned_text = clean_text(text)
	clean_text_vaccin_france.append(cleaned_text)



# perform sentiment analysis 

print('\n\nRQ 2 - Sentiment of vaccine-related tweets:')

sentiments_britany = []

for tweet in clean_text_vaccin_britany:
	valence = TextBlob(tweet, pos_tagger = PatternTagger(), analyzer = PatternAnalyzer()).sentiment[0]
	sentiments_britany.append(valence)

print('\nBritany:')
print(str(tweets_britany_df['vaccin'].value_counts()[1]) + ' tweets have mean valence of ' + str(np.mean(sentiments_britany)) + ' (sd=' + str(np.std(sentiments_britany)) + ')') 
print('Test for normality: stat=' + str(normaltest(sentiments_britany)[0]) + ', pval=' + str(normaltest(sentiments_britany)[1]))

sentiments_bouches = []

for tweet in clean_text_vaccin_bouches:
	valence = TextBlob(tweet, pos_tagger = PatternTagger(), analyzer = PatternAnalyzer()).sentiment[0]
	sentiments_bouches.append(valence)

print('\nBouches-du-Rhône:')
print(str(tweets_bouches_df['vaccin'].value_counts()[1]) + ' tweets have mean valence of ' + str(np.mean(sentiments_bouches)) + ' (sd=' + str(np.std(sentiments_bouches)) + ')')
print('Test for normality: stat=' + str(normaltest(sentiments_bouches)[0]) + ', pval=' + str(normaltest(sentiments_bouches)[1]))

sentiments_france = []

for tweet in clean_text_vaccin_france:
	valence = TextBlob(tweet, pos_tagger = PatternTagger(), analyzer = PatternAnalyzer()).sentiment[0]
	sentiments_france.append(valence)

print('\nFrance (control):')
print(str(tweets_france_df['vaccin'].value_counts()[1]) + ' tweets have mean valence of ' + str(np.mean(sentiments_france)) + ' (sd=' + str(np.std(sentiments_france)) + ')')
print('Test for normality: stat=' + str(normaltest(sentiments_france)[0]) + ', pval=' + str(normaltest(sentiments_france)[1]))



# statistical tests for sentiment analysis (hypothesis 2, sentiment as continuous)

print('\nSentiment as a continuous variable:')

stat, pval = ranksums(sentiments_britany, sentiments_bouches)
effsize = pg.compute_effsize(sentiments_britany, sentiments_bouches, eftype='CLES')

print('\nComparing Britany & Bouches-du-Rhône:')
print('stat=' + str(stat) + ', pval=' + str(pval) + ', CLES=' + str(effsize))

stat, pval = ranksums(sentiments_britany, sentiments_france)
effsize = pg.compute_effsize(sentiments_britany, sentiments_france, eftype='CLES')

print('\nComparing Britany & France (control):')
print('stat=' + str(stat) + ', pval=' + str(pval) + ', CLES=' + str(effsize))

stat, pval = ranksums(sentiments_bouches, sentiments_france)
effsize = pg.compute_effsize(sentiments_bouches, sentiments_france, eftype='CLES')

print('\nComparing Bouches-du-Rhône & France (control):')
print('stat=' + str(stat) + ', pval=' + str(pval) + ', CLES=' + str(effsize))



# show graphs for sentiment analysis, continuous variable (hypothesis 2)

plt.hist(sentiments_britany, weights=np.ones(len(sentiments_britany)) / len(sentiments_britany)*100, bins = 20, color = 'orange', alpha=0.75)
plt.title("Sentiment Analysis of vaccine-related tweets of Britany", fontsize = 18, fontweight = 'bold')
plt.xlabel("Sentiment Score", fontsize = 12, fontweight = 'bold')
plt.ylabel("Proportion of tweets in %", fontsize = 12, fontweight = 'bold')
plt.xlim([-1, 1])
plt.show()

plt.hist(sentiments_bouches, weights=np.ones(len(sentiments_bouches)) / len(sentiments_bouches)*100, bins = 20, color = 'green', alpha=0.75)
plt.title("Sentiment Analysis of vaccine-related tweets of Bouches-du-Rhône", fontsize = 18, fontweight = 'bold')
plt.xlabel("Sentiment Score", fontsize = 12, fontweight = 'bold')
plt.ylabel("Proportion of tweets in %", fontsize = 12, fontweight = 'bold')
plt.xlim([-1, 1])
plt.show()

plt.hist(sentiments_france, weights=np.ones(len(sentiments_france)) / len(sentiments_france)*100, bins = 20, color = 'red', alpha=0.75)
plt.title("Sentiment Analysis of vaccine-related tweets of France", fontsize = 18, fontweight = 'bold')
plt.xlabel("Sentiment Score", fontsize = 12, fontweight = 'bold')
plt.ylabel("Proportion of tweets in %", fontsize = 12, fontweight = 'bold')
plt.xlim([-1, 1])
plt.show()


plt.hist(sentiments_bouches, weights=np.ones(len(sentiments_bouches)) / len(sentiments_bouches)*100, bins = 20, color = 'green', alpha=0.3)
plt.hist(sentiments_france, weights=np.ones(len(sentiments_france)) / len(sentiments_france)*100, bins = 20, color = 'red', alpha=0.3)
plt.hist(sentiments_britany,  weights=np.ones(len(sentiments_britany)) / len(sentiments_britany)*100, bins = 20, color = 'orange', alpha=0.3)
plt.title("Sentiment Analysis of vaccine-related tweets by region", fontsize = 18, fontweight = 'bold')
plt.xlabel("Sentiment Score", fontsize = 12, fontweight = 'bold')
plt.ylabel("Proportion of tweets in %", fontsize = 12, fontweight = 'bold')
plt.xlim([-1, 1])
plt.legend(["Bouches-du-Rhône", "France", "Britany"])
plt.show()



# get categorical sentiment analysis: negative (-1) if score <= -0.2, positive (1) if score >= 0.2, neutral (0) otherwise

print('\nSentiment as a categorical variable:')

positivity_britany = []

for value in sentiments_britany: 
    if value >= 0.2:
        positivity_britany.append(1)
    elif value <= -0.2: 
        positivity_britany.append(-1)
    else : 
        positivity_britany.append(0)

positivity_bouches = []

for value in sentiments_bouches: 
    if value >= 0.2:
        positivity_bouches.append(1)
    elif value <= -0.2: 
        positivity_bouches.append(-1)
    else : 
        positivity_bouches.append(0)

positivity_france = []

for value in sentiments_france: 
    if value >= 0.2:
        positivity_france.append(1)
    elif value <= -0.2: 
        positivity_france.append(-1)
    else : 
        positivity_france.append(0)


pos_cat_britany = dict((x,positivity_britany.count(x)) for x in set(positivity_britany))
prop_neu_cat_britany = pos_cat_britany[0]/len(sentiments_britany)*100
prop_pos_cat_britany = pos_cat_britany[1]/len(sentiments_britany)*100
prop_neg_cat_britany = pos_cat_britany[-1]/len(sentiments_britany)*100

pos_cat_bouches = dict((x,positivity_bouches.count(x)) for x in set(positivity_bouches))
prop_neu_cat_bouches = pos_cat_bouches[0]/len(sentiments_bouches)*100
prop_pos_cat_bouches = pos_cat_bouches[1]/len(sentiments_bouches)*100
prop_neg_cat_bouches = pos_cat_bouches[-1]/len(sentiments_bouches)*100

pos_cat_france = dict((x,positivity_france.count(x)) for x in set(positivity_france))
prop_neu_cat_france = pos_cat_france[0]/len(sentiments_france)*100
prop_pos_cat_france = pos_cat_france[1]/len(sentiments_france)*100
prop_neg_cat_france = pos_cat_france[-1]/len(sentiments_france)*100

print('\nBritany:')
print('Positive tweets: ' + str(pos_cat_britany[1]) + ' out of ' + str(len(sentiments_britany)) + ' tweets (' + str(prop_pos_cat_britany) + ' %)')
print('Neutral tweets: ' + str(pos_cat_britany[0]) + ' out of ' + str(len(sentiments_britany)) + ' tweets (' + str(prop_neu_cat_britany) + ' %)')
print('Negative tweets: ' + str(pos_cat_britany[-1]) + ' out of ' + str(len(sentiments_britany)) + ' tweets (' + str(prop_neg_cat_britany) + ' %)')

print('\nBouches-du-Rhône:')
print('Positive tweets: ' + str(pos_cat_bouches[1]) + ' out of ' + str(len(sentiments_bouches)) + ' tweets(' + str(prop_pos_cat_bouches) + ' %)')
print('Neutral tweets: ' + str(pos_cat_bouches[0]) + ' out of ' + str(len(sentiments_bouches)) + ' tweets (' + str(prop_neu_cat_bouches) + '%)')
print('Negative tweets: ' + str(pos_cat_bouches[-1]) + ' out of ' + str(len(sentiments_bouches)) + ' tweets(' + str(prop_neg_cat_bouches) + '%)')

print('\nFrance:')
print('Positive tweets: ' + str(pos_cat_france[1]) + ' out of ' + str(len(sentiments_france)) + ' tweets(' + str(prop_pos_cat_france) + ' %)')
print('Neutral tweets: ' + str(pos_cat_france[0]) + ' out of ' + str(len(sentiments_france)) + ' tweets (' + str(prop_neu_cat_france) + ' %)')
print('Negative tweets: ' + str(pos_cat_france[-1]) + ' out of ' + str(len(sentiments_france)) + ' tweets(' + str(prop_neg_cat_france) + ' %)')



# show graphs for sentiment analysis, categorical variable (hypothesis 2)

positivity_all = [{'Region': 'Britany', '-1': prop_neg_cat_britany, '0': prop_neu_cat_britany, '1': prop_pos_cat_britany}, 
					{'Region':'Bouches-du-Rhône',  '-1': prop_neg_cat_bouches, '0': prop_neu_cat_bouches, '1': prop_pos_cat_bouches}, 
					{'Region':'France (control)', '-1': prop_neg_cat_france, '0': prop_neu_cat_france, '1': prop_pos_cat_france}]
positivity_all_df = pd.DataFrame(positivity_all)

positivity_all_df.plot.bar(x = 'Region', rot = 0, y = ['-1', '0', '1'], capsize = 10, color = ['mediumseagreen', 'darkcyan', 'cornflowerblue'], linewidth =  0.2, edgecolor = 'black', )
plt.title('Sentiment Analysis (categorical) of vaccine-related tweets by region 1', fontsize = 18, fontweight = 'bold')
plt.xlabel("Region", fontsize = 12, fontweight = 'bold')
plt.ylabel("Vaccine-related Tweets (in %)", fontsize = 12, fontweight = 'bold')
plt.show()


positivity_all2 = [{'Region': 'Britany', '-1': prop_neg_cat_britany, '1': prop_pos_cat_britany}, 
					{'Region':'Bouches-du-Rhône',  '-1': prop_neg_cat_bouches, '1': prop_pos_cat_bouches}, 
					{'Region':'France (control)', '-1': prop_neg_cat_france, '1': prop_pos_cat_france}]
positivity_all2_df = pd.DataFrame(positivity_all2)

positivity_all2_df.plot.bar(x = 'Region', rot = 0, y = ['-1', '1'], capsize = 10, color = ['mediumseagreen', 'cornflowerblue'], linewidth =  0.2, edgecolor = 'black', )
plt.title('Sentiment Analysis (categorical) of vaccine-related tweets by region 2', fontsize = 18, fontweight = 'bold')
plt.xlabel("Region", fontsize = 12, fontweight = 'bold')
plt.ylabel("Vaccine-related Tweets (in %)", fontsize = 12, fontweight = 'bold')
plt.show()



# statistical tests for sentiment analysis (hypothesis 2, sentiment as categorical)

obs = np.array([
	[pos_cat_britany[1],pos_cat_britany[0],pos_cat_britany[-1]],
	[pos_cat_bouches[1],pos_cat_bouches[0],pos_cat_bouches[-1]],
	[pos_cat_france[1],pos_cat_france[0],pos_cat_france[-1]]
	])

stat, pval, dof, expd = chi2_contingency(obs)
n = np.sum(obs)
minDim = min(obs.shape)-1
V = np.sqrt((stat/n)/minDim)

print('\nComparing all 3 regions:')
print('stat=' + str(stat) + ', pval=' + str(pval) + ', V=' + str(V))



print('\n')
