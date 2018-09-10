'''
This file contains functions and variables
used in eda.py file and the P4_EDA.ipynb notebook

INDEX:

amnits_catgry_dicc:
Variable to map the amenities into determined categories

amnits_catgry_labels:
Variable to map the amenities crude labels to their correct labels


breakpoints
Variable to assign the price into broader intervals.


'''
import numpy as np
import pandas as pd
from bisect import bisect

# Used only in notebook
import pandas_profiling
import missingno as msn
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
# Update matplotlib defaults to something nicer
plt_update = {'font.size':20,
              'xtick.labelsize':14,
              'ytick.labelsize':14,
              'figure.figsize':[10.0,5.0],
              'axes.labelsize':20,
              'axes.titlesize':20,
              'lines.linewidth':3}
sns.set(style="darkgrid", color_codes=True)
plt.rcParams.update(plt_update)

'''------------------PRE-PROCESSING FUNCTIONS----------------'''

''' NOT USED
Function getting a zipcode as a parameter
It gathers all the listings under that zipcode and performs
a 1 cluster k means with the listings latitude and longitude
and returns the centroid'''
def get_centroid(zipc):
    zb = mybar_data.loc[mybar_data.zipcode==zipc, ['latitude','longitude']].as_matrix()
    # Perform K-Means
    kmeansLatLon = KMeans(n_clusters=1, max_iter=10, n_init=1, init='random').fit(zb)
    # Get the centroid
    return kmeansLatLon.cluster_centers_


'''Function that reduces the number of unique values into larger intervals
    In this specific case, it deals with price.
'''
def get_interval(total):
    return breakpoints[bisect(breakpoints, total)-1]


'''Loads the reviews.csv file from the Airbnb site and counts the number of
    reviews per month per listing and returns the average number of
    reviews per month per listing in form of a dataframe.
'''
def get_reviews_per_month():
    # Get the number of reviews per month per listing
    rpm = reviews_data.groupby(['listing_id','date']).agg({'id': 'count'}).unstack(level=0).resample("M", how='sum').stack().reset_index()
    # Get rid of the null rows
    rpm = rpm.loc[rpm.id!=0]
    # Rename id to reviews per month
    rpm.rename(columns={'id':'review_per_month'}, inplace=True)
    # Now group the df by listing and get the average of the review per month
    return rpm.groupby('listing_id').mean()

'''------------------PRE-PROCESSING VARIABLES----------------'''

'''Breakpoints to map the sparse price'''
breakpoints = [20, 40, 60, 80, 100, 120, 160, 200, 300, 500, 800, 1000, 5000, 10000]

'''Current necessary columns'''
host_col = ['listing_id','host_id', 'host_since',
       'host_response_time', 'host_neighbourhood', 'host_listings_count',
       'host_verifications', 'host_is_superhost']
loc_col = ['city', 'zipcode',
       'latitude', 'longitude']
prop_col = ['property_type', 'room_type', 'accommodates',
       'bathrooms', 'beds', 'bed_type', 'amenities', 'price',
       'guests_included', 'extra_people', 'minimum_nights', 'maximum_nights',
       'availability_365', 'instant_bookable', 'cancellation_policy']
guest_col = ['require_guest_profile_picture', 'require_guest_phone_verification']
rvw_col = ['number_of_reviews', 'first_review', 'last_review',
       'review_scores_rating', 'review_scores_accuracy',
       'review_scores_cleanliness', 'review_scores_checkin',
       'review_scores_communication', 'review_scores_location',
       'review_scores_value', 'reviews_per_month']

'''Data prior to EDA'''
int_data = '../data/processed/data.csv'
usecols = host_col+loc_col+prop_col+guest_col+rvw_col
mybar_data = pd.read_csv(int_data, index_col=1, parse_dates=['host_since','first_review','last_review'], usecols=usecols)

int_data = '../data/raw/paris_reviews.csv'
reviews_data = pd.read_csv(int_data, parse_dates=['date'], usecols=['listing_id','date','id'])

#---------------------- EDA FUNCTIONS -----------------------

'''
get_amenity_tools takes a dataframe with listings as index
and a column of a list of its amenities
the amenities per listing and returns:

- my_data, a dataframe holding the amenities for each listing in its original
form (between keys:{}) and in form of a list
and a third colum with the total number of amenities per listing.

- prop_amnits, a df with a list of unique amenities with columns defining the
total number in the dataset as well as the amenity's category

- amnits_ctgry, a df with the available amenities' categories with columns
calculating the category's size and normalized size.

-
'''
def get_amenity_tools(my_data, n_listings):

    my_data['total_amenities'] = my_data.amnty.apply(lambda x: len(x))

    ## CREATE A DATAFRAME HOLDING EACH AMENITY AND IT's TOTAL OCCURRENCE FOR EVERY LISTING
    amnits = {}
    for amenity_listing in my_data.amnty:
        #print(amenity_listing, '\n')
        for amenty in amenity_listing:
            if amenty in amnits.keys():
                amnits[amenty]+=1
            else:
                amnits[amenty] = 1

    # Convert dictionary to DataFrame (prop_amnits):
    prop_amnits = pd.DataFrame(list(amnits.items()), columns=['Amnty','Count'])

    ### MAP AMENITIES TO THEIR PRE-DETERMINED (by me) CATEGORY
    # Inverse the amnits_catgry_dicc dictionary ('Amenity':'Category abbv') in order to map the amenities to their respective category.
    amnty_ctgry_mapping = {i:k for k, v in amnits_catgry_dicc.items() for i in v}

    prop_amnits['Category'] = prop_amnits.Amnty.map(amnty_ctgry_mapping)
    # Map the category crude label to the normal ones.
    prop_amnits['Category'] = prop_amnits.Category.map(amnits_catgry_labels)
    prop_amnits['Proportion'] = prop_amnits.Count.apply(lambda x: x/n_listings*100).astype(int)


    # CREATE A DATAFRAME WITH LISTING EVERY CATEGORY ALONG WITH THE NUMBER OF OCCURRENCES OF ITS ELEMENTS
    # NORMALIZE THE NUMBER AND ALSO INCLUDE THE NUMBER OF ELEMENTS EACH CATEGORY HAS.

    amnits_ctgry = prop_amnits.loc[:,['Count','Category']].groupby('Category').sum().sort_values('Count')
    # Normalize the Count
    amnits_ctgry['nCount'] = amnits_ctgry.apply(lambda x: (x-amnits_ctgry.Count.min())/(amnits_ctgry.Count.max() - amnits_ctgry.Count.min()))

    # Dictionary that has every category and it's size in elements.
    num_amnits_per_cat = {amnits_catgry_labels[k]:len(v) for k, v in amnits_catgry_dicc.items()}

    ## Add to the dataframe amnits_ctgry the size of each category
    amnits_per_cat = amnits_ctgry.reset_index()
    amnits_per_cat['size_ctgry'] = amnits_per_cat.Category.map(num_amnits_per_cat)
    amnits_ctgry = pd.merge(amnits_ctgry, amnits_per_cat.set_index('Category').loc[:,['size_ctgry']], right_index=True, left_index=True)

    # Normalize the size_ctgry so that it can fit in the same graph
    amnits_ctgry['nSize'] = amnits_ctgry.size_ctgry.apply(lambda x: (x/amnits_ctgry.size_ctgry.sum()))

    return my_data, prop_amnits, amnits_ctgry


#---------------------- EDA VARIABLES -----------------------


'''Variable to map the amenities crude labels to their correct labels'''
amnits_catgry_labels = {
'hy_n_bthrm': 'Hygiene and bathroom',
'fd_n_ktchn': 'Food and Kitchen',
'bdrm': 'Bedroom',
'frntre': 'Furniture',
'sfty_n_scrty': 'Safety and Security',
'pts': 'Pets',
'elctrncs_n_entrtnmnt': 'Electronics and Entertainment',
'accssblty': 'Accessibility',
'outdrs': 'Outdoors',
'othr': 'Other'}

'''Variable to map the amenities into determined categories'''

amnits_catgry_dicc = {
'hy_n_bthrm': ['toilet','Accessible-height toilet','Baby bath','Baby monitor',
'Babysitter recommendations','Bath towel','Bathroom essentials','Bathtub',
'Bathtub with bath chair','Body soap','En suite bathroom','Fixed grab bars for shower',
'Fixed grab bars for toilet','Hair dryer','Handheld shower head','Hangers',
'Hot tub','Hot water','Private bathroom','Rain shower','Roll-in shower',
'Shampoo','Shower chair','Toilet paper','Wide clearance to shower'],
'fd_n_ktchn': ['Breakfast' , 'Breakfast table' , 'Coffee maker' , 'Cooking basics' ,
'Children’s dinnerware' , 'Dishes and silverware' ,
'Dishwasher' , 'Espresso machine' , 'Full kitchen' , 'Hot water kettle' , 'Kitchen' , 'Kitchenette' ,
'Microwave' , 'Mini fridge' , 'Oven' , 'Refrigerator' , 'Stove'],
'bdrm': ['Bed','Accessible-height bed', 'Bed linens','Bedroom comforts','Crib','Electric profiling bed',
 'Firm mattress','Extra pillows and blankets','Pillow-top mattress','Room-darkening shades'],
'frntre': ['High chair' , 'Changing table' , 'Pack ’n Play/travel crib' ,
'Indoor fireplace','Outlet covers'],
'sfty_n_scrty': ['Carbon monoxide detector','Building staff','Doorman',
                'Buzzer/wireless intercom','Fire extinguisher','Fireplace guards',
                'Private entrance','Private living room','First aid kit',
                'Lock on bedroom door','Lockbox','Safety card','Smoke detector',
                'Table corner guards','Smart lock','Window guards'],
'pts': ['Cat(s)','Dog(s)','Other pet(s)' , 'Pets allowed' , 'Pets live on this property'],
'elctrncs_n_entrtnmnt': ['Air conditioning','Air purifier', 'Dryer','Heating','Iron',
'TV','Washer','Washer / Dryer','Wifi',
'Cable TV','Laptop friendly workspace','Internet','Pocket wifi',
'Children’s books and toys','Ethernet connection','Game console','EV charger'],
'outdrs': ['BBQ grill', 'Balcony', 'Garden or backyard' ,
'Beach essentials' , 'Beachfront' , 'Gym' , 'Patio or balcony' , 'Pool' ,
'Ski in/Ski out' , 'Waterfront'],
'accssblty': ['Disabled parking spot','Elevator','Flat path to front door',
            'Free parking on premises','Paid parking off premises','Paid parking on premises',
            'Free street parking','Ground floor access','Keypad','Lake access',
            'Well-lit path to entrance','Wheelchair accessible','Single level home',
            'Step-free access','Stair gates','Wide doorway','Wide clearance to bed',
            'Wide entryway','Wide hallway clearance','Host greets you','24-hour check-in',
            'Self check-in'],
'othr': ['translation missing: en.hosting_amenity_49','translation missing: en.hosting_amenity_50',
'Essentials','Other',' ','Cleaning before checkout','Family/kid friendly',
'Long term stays allowed','Luggage dropoff allowed',
'Smoking allowed','Suitable for events']}
