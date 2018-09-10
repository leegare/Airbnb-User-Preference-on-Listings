'''
This file contains functions and variables
used in pre_processing.py file and the pre_processing notebook
'''

import numpy as np
import pandas as pd
import re

# Used only in notebook
import pandas_profiling
import missingno as msn
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')
# Update matplotlib defaults to something nicer
plt_update = {'font.size':16,
              'xtick.labelsize':14,
              'ytick.labelsize':14,
              'figure.figsize':[10.0,5.0],
              'axes.labelsize':20,
              'axes.titlesize':20,
              'lines.linewidth':3}
sns.set(style="darkgrid", color_codes=True)
plt.rcParams.update(plt_update)

'''------------------PRE-PROCESSING FUNCTIONS----------------'''

'''Takes a zipcode and returns the most common city assigned to the zipcode'''
def get_com_den_city(z, zip_data):
    if isinstance(zip_data.loc[z,'city'], str):
        return zip_data.loc[z,'city'].lower()
    else:
        com_den = zip_data.loc[z,'city'].value_counts()
        for city_value in com_den.index:
            if city_value.lower() != 'paris':
                return city_value.lower()


'''USED IN NOTEBOOK ONLY
Function that takes 2 strings comparing them and outputs a 1 if the
first string matches at least the first 3 words the second string has'''
def sum_desc_comp(s1, s2):
    if s2.find(s1) != -1:
        return 1
    else:
        return 0

'''USED IN NOTEBOOK ONLY
Quantify host_response_time
    Function that receives a string of either:
        - within an hour
        - within a few hours
    ...
    and so and converts (grades) it into a number.
'''
def quantify_host_response_time(T):
    if (T == 'within an hour'):
        return 5
    elif (T == 'within a few hours'):
        return 10
    elif (T == 'within a day'):
        return 15
    else:
        return 20

'''Selecting numeric-only string zipcodes:
    Function that takes in a string (or a float)
    strips it out of non-numerical characters
    and returns it as an int of 5 digits
'''
def clean_zip(z1):
    zip_code_pattern = '\d{5}'
    uneccesary_chars = '[\r\n]?[\s-]?[aA-zZ]?'
    if isinstance(z1, float) or isinstance(z1, int):
        z1 = str(z1)
    try:
        new_zip = re.sub(uneccesary_chars, r"", z1)
        if re.search(zip_code_pattern, new_zip):
            return int(new_zip[:5])
        else:  # i.e '1446.0'
            return 0
    except:
        print(type(z1),'is a loophole, or a:', z1)
        return 1


'''------------------PRE-PROCESSING VARIABLES----------------'''

'''Columns after dataset's 2nd filter'''
columns_phase_2 = {
'summry':['name','description','last_scraped'],
'host_info':['host_id',
'host_since',
'host_about',
'host_response_time',
'host_response_rate',
'host_neighbourhood',
'host_listings_count',
'host_verifications',
'host_is_superhost'],
'location':['city','zipcode','latitude','longitude'],
'prdct':['property_type',
 'room_type',
 'accommodates',
 'bathrooms',
 'beds',
 'bed_type',
 'amenities',
 'price', 'weekly_price', 'monthly_price',
 'guests_included',
 'extra_people',
 'minimum_nights',
 'maximum_nights',
 'availability_30',
 'availability_60',
 'availability_90',
 'availability_365',
 'instant_bookable',
 'cancellation_policy'],
'guest_info':['require_guest_profile_picture', 'require_guest_phone_verification'],
'rvws':['number_of_reviews',
 'first_review',
 'last_review',
 'review_scores_rating',
 'review_scores_accuracy',
 'review_scores_cleanliness',
 'review_scores_checkin',
 'review_scores_communication',
 'review_scores_location',
 'review_scores_value',
 'reviews_per_month']}


'''Columns after dataset's 1st observation'''
columns_phase_1 = {
'headr':['last_scraped'],
'summry':['name',
 'summary',
 'space',
 'description',
 'neighborhood_overview',
 'notes',
 'transit',
 'access',
 'interaction',
 'house_rules'],
'host_info':['host_id',
 'host_since',
 'host_location',
 'host_name',
 'host_about',
 'host_response_time',
 'host_response_rate',
 'host_neighbourhood',
 'host_listings_count',
 'host_verifications', 'host_is_superhost'],
'location':['city',
 'zipcode',
 'smart_location',
 'latitude',
 'longitude',
 'is_location_exact'], # To remove later
'prdct':['property_type',
 'room_type',
 'accommodates',
 'bathrooms',
 'beds',
 'bed_type',
 'amenities',
 'price', 'weekly_price', 'monthly_price',
 'guests_included',
 'extra_people',
 'minimum_nights',
 'maximum_nights',
 'availability_30',
 'availability_60',
 'availability_90',
 'availability_365',
 'instant_bookable',
 'cancellation_policy'],
'guest_info':['require_guest_profile_picture', 'require_guest_phone_verification'],
'rvws':['number_of_reviews',
 'first_review',
 'last_review',
 'review_scores_rating',
 'review_scores_accuracy',
 'review_scores_cleanliness',
 'review_scores_checkin',
 'review_scores_communication',
 'review_scores_location',
 'review_scores_value',
 'reviews_per_month']}

'''Original columns'''
columns_phase_0 = {
'headr':['listing_url', 'scrape_id', 'last_scraped'],
'summry':['name', 'summary',
       'space', 'description', 'experiences_offered', 'neighborhood_overview',
       'notes', 'transit', 'access', 'interaction', 'house_rules',
       'thumbnail_url', 'medium_url', 'picture_url', 'xl_picture_url'],
'host_info':['host_id', 'host_url', 'host_name', 'host_since', 'host_location',
       'host_about', 'host_response_time', 'host_response_rate',
       'host_acceptance_rate', 'host_is_superhost', 'host_thumbnail_url',
       'host_picture_url', 'host_neighbourhood', 'host_listings_count',
       'host_total_listings_count', 'host_verifications',
       'host_has_profile_pic', 'host_identity_verified'],
'location':['street','neighbourhood', 'neighbourhood_cleansed',
       'neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market',
       'smart_location', 'country_code', 'country', 'latitude', 'longitude',
       'is_location_exact'],
'prdct':['property_type', 'room_type', 'accommodates',
       'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'square_feet',
       'price', 'weekly_price', 'monthly_price', 'security_deposit',
       'cleaning_fee', 'guests_included', 'extra_people', 'minimum_nights',
       'maximum_nights', 'calendar_updated', 'has_availability',
       'availability_30', 'availability_60', 'availability_90',
       'availability_365','instant_bookable','cancellation_policy', 'requires_license',
       'license', 'jurisdiction_names','is_business_travel_ready'],
'guest_info':['require_guest_profile_picture', 'require_guest_phone_verification'],
'rvws':['calendar_last_scraped', 'number_of_reviews',
       'first_review', 'last_review', 'review_scores_rating',
       'review_scores_accuracy', 'review_scores_cleanliness',
       'review_scores_checkin', 'review_scores_communication',
       'review_scores_location', 'review_scores_value', 'reviews_per_month']}
