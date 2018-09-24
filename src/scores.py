# MAIN SCRIPT

#from analysis_funk import * # Stored functions and variables to plot the map of Paris
from mvp_funk import *


'''
1. occupancy_scores.occupancy_rate refers to the score defined by the reviews per month
2. qlty.listings_review_score refers to the score defined by the quality of the listing in terms of its predefined parameters
and
3. myloc.location_score refers to the score defined by the amount of attractions the listing is near to.
'''



## LOCATION SCORE

print('Calculating location score ..')

myloc = data.loc[:,['zipcode','latitude','longitude','price','review_scores_location']].sample(frac=1)

def get_distEuclid_from_attractions(x):
    d = np.sqrt((paris_attractions.longitude - x.longitude)**2 + (paris_attractions.latitude - x.latitude)**2)
    #return [d.sort_values()] # as a list of a serie
    return [d.values] # As a list of an array

# Get euclidean distance
myloc['attraction'] = myloc.apply(lambda x: get_distEuclid_from_attractions(x), axis=1)

# Transform list of distances into dataframe along with its listing id.
all_dist = []
ndist = []
number_attr = paris_attractions.shape[0]
for idx, content in myloc.iterrows():
    all_dist+=[list(np.repeat(idx,number_attr)),list(content.attraction[0][:number_attr])]
idx = []
for i in all_dist[0:-1:2]:
    idx+=i
for d in all_dist[1::2]:
    ndist+=d
nDist = pd.DataFrame({'listing' : idx, 'distance': ndist})
myloc.drop('attraction', axis=1, inplace=True)

# Get all listings' distance to their closest neighbor
closest_neighbor_distances = nDist.groupby('listing').min()

# Take the longest distance from that
top_dist = closest_neighbor_distances.loc[closest_neighbor_distances.idxmax().values[0]].distance

# Get all the listings with their distances below the top_dist
location_listings = nDist.loc[nDist.distance<=top_dist].groupby('listing').count().rename(columns={'distance':'num_of_attractions'}).sort_values('num_of_attractions', ascending=False)

# Grade the listings with a normalized metric reflecting the number of attractions.
myloc['location_score'] = location_listings.num_of_attractions.apply(lambda x: (x-min(location_listings.num_of_attractions))/(max(location_listings.num_of_attractions)-min(location_listings.num_of_attractions)))

# Build dataframe with the number of listings having n number of attractions within the radius of top_dist
num_atrr_vs_num_list = location_listings.reset_index().groupby('num_of_attractions').count()

# Normalize review_scores_location
myloc['review_location'] = (myloc.review_scores_location-min(myloc.review_scores_location))/(max(myloc.review_scores_location)-min(myloc.review_scores_location))

# Merge location_score with review_scores_location
myloc['location_score'] = myloc.loc[:,['location_score','review_location']].mean(axis=1)



## OCCUPANCY RATE


print('Calculating occupancy rate ..')

occupancy_scores = data.loc[:,['reviews_per_month', 'zipcode','longitude','latitude']].sample(frac=1)
occupancy_scores.rename(columns={'reviews_per_month':'occupancy_rate'}, inplace=True)

# Get the limit from which the remaining points are considered outliers.
q1 = occupancy_scores.occupancy_rate.quantile(0.25)
q3 = occupancy_scores.occupancy_rate.quantile(0.75)
_iqr = q3-q1
outlier_mark=q3+1.5*_iqr

occupancy_scores.occupancy_rate = occupancy_scores.occupancy_rate.apply(lambda x: x if x<outlier_mark else outlier_mark)

# Normalize:
m = occupancy_scores.occupancy_rate.min()
M = occupancy_scores.occupancy_rate.max()
occupancy_scores.occupancy_rate = occupancy_scores.occupancy_rate.apply(lambda x: (x-m)/(M-m))



## LISTING'S REVIEW RATE


print("Calculating listing's review rate ..")

q_col = ['host_is_superhost','room_type','bed_type','bathrooms','amty_per','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_value']
qlty = data.loc[:,q_col+['zipcode','longitude','latitude']].sample(frac=1)

# Set host_is_superhost values from 0 to 1
qlty.host_is_superhost = qlty.host_is_superhost.apply(lambda x: x)
# Map room type
rt_map = {'Entire home/apt': 1, 'Private room': 0.5, 'Shared room':0}
qlty.room_type = qlty.room_type.map(rt_map)
# Map bed type
bt_map = {'Real Bed': 1, 'Pull-out Sofa': 0.5, 'Airbed':0.5, 'Couch':0, 'Futon':0}
qlty.bed_type = qlty.bed_type.map(bt_map)
# Normalize amty_per
qlty.amty_per = qlty.amty_per.apply(lambda x: x/100)
# Normalize bathrooms
qlty.bathrooms = qlty.bathrooms.apply(lambda x: (x-qlty.bathrooms.min())/(qlty.bathrooms.max()-qlty.bathrooms.min()))
# Normalize review scored by 100
qlty.review_scores_accuracy = qlty.review_scores_accuracy.apply(lambda x: x/10)
qlty.review_scores_cleanliness = qlty.review_scores_cleanliness.apply(lambda x: x/10)
qlty.review_scores_checkin = qlty.review_scores_checkin.apply(lambda x: x/10)
qlty.review_scores_communication = qlty.review_scores_communication.apply(lambda x: x/10)
qlty.review_scores_value = qlty.review_scores_value.apply(lambda x: x/10)
qlty['listings_review_score'] = qlty[q_col].mean(axis=1)


print('Scores retrieval completed!')

# Merge results : Nested merge
scores = pd.merge(pd.merge(qlty.loc[:,['zipcode','longitude','latitude','listings_review_score']], occupancy_scores.loc[:,['occupancy_rate']], right_index=True, left_index=True), myloc.loc[:,['location_score']], right_index=True, left_index=True)

scores.to_csv('../data/processed/scores.csv')
