import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bisect import bisect
import warnings
warnings.filterwarnings('ignore')

pd.set_option('precision', 3)


'''----------------Global Variables--------------------------'''
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

orate = {0.4:5, 0.6:15, 0.8:20, 0.9:25, 0.925:35, 0.95:45, 0.975:55, 1:75}

'''---------------------FUNCTIONS---------------------'''

'''Function that takes the number of bookings per year times the minimum
number of nights of a listing and returns it
divided by the number of days available. Basically it captures the
occupancy_rate
'''
def get_occupancy_rate(r):

    number_of_months = 12
    min_nights = 4.5
    booking_per_year = r.reviews_per_month * number_of_months
    days_available = r.availability_365

    if days_available == 0: # For those with 0 availability
        days_available = 1

    #    print(days_available/(booking_per_year)>=n_nights,'Listing with',days_available,'/365 days available had',
    #      r.booking_per_year,'guests, staying for a max of',days_available/booking_per_year,'nights,
    #      so rate is:',occupancy_rate)

    return min(1, (min_nights*booking_per_year)/days_available)


'''Function that reduces the number of unique values into larger intervals'''
def get_interval_on_rate(total):
    return orate[bisect(orate, total)-1]
