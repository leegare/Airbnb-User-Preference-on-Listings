## Script

# IMPORT VARIABLES, LIBRARIES AND FUNCTIONS

from eda_funk import * # Stored functions and variables regarding EDA analysis
from analysis_funk import * # Stored functions and variables to plot the map of Paris
print('EDA loading')
data = mybar_data

# Local function
'''Function that needs a global variable with the essential amenities
and it returns how much (%) of those amenities each listing has.
'''
def get_essential_amenities_percentage(x):
    p = 0.0
    for amty in listings_w_essentials_top_list.Amnty.values:
        if amty in x:
            p+=1
    return p/len(listings_w_essentials_top_list.Amnty.values)*100


# Get the amenities per listing
my_amnty = data.loc[:,['amenities']].sample(frac=1)
my_amnty['amnty'] = my_amnty.amenities.apply(lambda x: x.replace('{','').replace('}','').replace('"','').split(','))
my_amnty.drop('amenities', axis=1, inplace=True)

# Get those listings where they checked in "Essentials" and those who didnt

essentials = my_amnty.amnty.apply(lambda x: 'Essentials' in x)
e_amenities_per_listing, e_amenity_df, e_amenity_cat = get_amenity_tools(my_amnty[essentials], my_amnty[essentials].shape[0])

# Remove the Amenity "Essentials" from the list as it's being taken into consideration.

listings_w_essentials_top_list = e_amenity_df.sort_values(['Proportion','Count'], ascending=False).head(17).loc[:,['Amnty','Proportion']]
listings_w_essentials_top_list = listings_w_essentials_top_list[listings_w_essentials_top_list.Amnty != 'Essentials'].reset_index()

# Get listings with their amenities and the percentage of essentials they have

my_amnty['amty_per'] = my_amnty.amnty.apply(get_essential_amenities_percentage)

# Merge to the main dataframe

data = pd.data = pd.merge(data, my_amnty, right_index=True, left_index=True)
data.drop('amenities', inplace=True, axis=1)
data.rename(columns={'amnty':'amenities'}, inplace=True)

# remove the listings with null price
data = data.loc[data.price>0]
# Get the accurate number of reviews per listing

review_count = reviews_data.loc[:,['listing_id','date']].groupby('listing_id').count()
review_count.rename(columns={'date':'num_reviews'}, inplace=True)
data = pd.merge(data, review_count, right_index=True, left_index=True)
data['num_rv_verify'] = data.number_of_reviews==data.num_reviews
data.loc[data.num_rv_verify==False,['number_of_reviews','num_reviews']].shape

data.drop(['number_of_reviews','num_rv_verify'], axis=1, inplace=True)
data.rename(columns={'num_reviews':'number_of_reviews'}, inplace=True)

# Verify the accuracy of the reviews_per_month

rpm = get_reviews_per_month()

# Compare if the reviews per month are equal

data = pd.merge(data, rpm, right_index=True, left_index=True)
data['check_rpm'] = data.reviews_per_month==data.review_per_month
data.loc[data.check_rpm==False,].shape

data.drop(['check_rpm','reviews_per_month'], axis=1, inplace=True)
data.rename(columns={'review_per_month':'reviews_per_month'}, inplace=True)
data.index.name = 'listing_id'


#### Tourist Attractions
# Get those attractions that are only in Paris

paris_at = a_data.loc[(a_data.latitude>48.813500)&((a_data.latitude<48.906000))&(a_data.longitude>2.25)&((a_data.longitude<2.421)),
                           ['Name','latitude','longitude']]

# Remove Specific outliers

paris_at = paris_at[~paris_at.Name.isin(specific_outliers)]
#inliers = r"canal|centre|parc|jardin|palace|place|stade|champ|quarter|musée|carré|chapel|centre|fontaine|château|fountain|museum|bateau|passerelle|bassin|pavillon|passage|abbey|cathedral|théâtre|tour|church|synagogue|chapelle|quartier|port|pont|square|palais|canal|centre|institut|café|cemetery|cimetière|île|hôtel|bois"
outlier_pattern = r"library|croix-rouge|arrondissement of paris$|\Sparis métro\S$|\Sparis rer\S$|\Sparis métro and rer\S$|minister|embassy|collège|prefecture|lycée|paristech|^mairie|hôpital|hospital|porte|hotel|school|école|avenue|gare|banque|show|university|college|bibliothèque|^rue|boulevard|café|fontaine|hôtel|institut|campus|passage|pont|stade|théâtre"
paris_at['valid_attraction'] = [False if re.search(outlier_pattern, n) else True for n in paris_at.Name]
paris_attractions = paris_at.loc[paris_at.valid_attraction == True, ['Name','latitude','longitude']].set_index('Name').sort_index().reset_index()


#Save
data.to_csv('../data/processed/data.csv')
paris_attractions.to_csv('../data/processed/paris_attractions.csv')
print("EDA completed. Data has been processed and saved in ../data/processed/")
