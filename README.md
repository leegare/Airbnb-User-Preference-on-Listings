# Airbnb: Hosting the location or the property? 
---
# Introduction

Tourism is a major business in pretty much every country who has places and experiences to offer and catch the interest of foreigners. Essential services for the tourists ranging from restaurants to hotels have flourished and we've seen the company Airbnb enter in a peculiar manner. 

In the old days when hotels were sparse, tourists had no choice of its location so they resorted to choose establishments with other appealing features. Then came Airbnb and democratized the location giving tourists the option to get places nearby the touristy attractions. 

I'd like to understand how tourists make choices involving hotel hunting and in order to have an accurate study a huge dataset is needed. A city with a myriad of listings has to be a city in the top ten places to visit and among them, there are no best candidates than Paris in France. The tourist capital of the world, with a breaking record 40 million tourists per year, Paris is prepared to welcome them. 

---

# Background

Having lived in Paris for 4 years I became a francophiliac. To avoid forgetting the beautiful language they speak I keep up with the actualities in France and I was getting updates when I stumbled with this video that caught my attention:(https://www.youtube.com/watch?v=Iywwa3wfhoU&t=35s). It's about Airbnb and its disruptive innovation that is affecting the residential housing in tourist-concentrated cities like Barcelona and Paris.

___
# *Are Airbnb guests more interested in listings that are in convenient neighbourhoods regardless of the condition of the property?*
---

# Questions and Goals

*To answer the above question, the **occupancy rate** of each listing needs to be calculated as it is the actual popularity rating and it's not provided by the dataset. Secondly this project will aim to answer the following questions:*

#### I. Where are the best listings? 
- What parameters define a popular listing? 

#### II. Which are Paris' convenient neighbourhoods to stay at?
- Define best/worst neighbourhoods with respect to the listing's price, proximity to transportation access, tourist attractions, parks and recreations. 
    - What is the proportion of high rated listings per district? 

---

# The Dataset

The data is taken from http://insideairbnb.com/get-the-data.html where the website offers popular city listings updated to this year. The data can be summarized in 6 categories: host, listing's summary, listing's specs, guest, location and reviews. 

Originally the dataset held 62848 listings described in 91 columns. These listings have been collected from 2009 to this year. 

The location information for listings are anonymized by Airbnb. In practice, this means the location (latitude and longitude) for a listing on the map, or in the data will be from 0-450 feet (150 metres) of the actual address.

Neighbourhood names for each listing are compiled by comparing the listing's geographic coordinates with a city's definition of neighbourhoods. Airbnb neighbourhood names are not used because of their inaccuracies.

---
# Initial Data Cleaning Approach and Exploratory Findings

From the 91 columns, I filtered out 30 columns containing paragraphs of descriptions and reviews and variables that were unnecessary (correlated) or irrelevant to the analysis. On the second filter, columns with a high volume of missing or incorrect values got removed leaving the dataset with 49 pertinent columns. I then grouped them in 5 categories: Host qualities, Listing specs, Location data and Review scores.  

## Cleaning listings coordinates

I found 3 types of innacuracies regarding the zipcodes, city name and coordinates. The zipcodes and the cities were either mispelled or unrelated. After standardizing the cities and zipcodes I grouped all the listings by zipcode and took the common city name to be the official name for the whole zipcode. The following figure shows the listings classified by color according to their zipcode.  

![**Figure PRE original listings per zipcode**](images/Centroids_and_original_listings.png)

An LOF process showed 208 listings with either an incorrect zipcode or coordinates. I had to assume that the coordinates were correct so I proceeded to adjust the zipcode according to its nearest zipcode center of mass as shown below: 

![**Figure PRE original listings per zipcode**](images/Centroids_and_final_listings.png)



---
## EDA: What are the factors that we can use to consider a good listing?

### *In host qualities:*

HOST_RESPONSE_TIME  

It's a categorical variable holding 4 unique values: 'within an hour', 'within a day', 'within a few hours','a few days or more'.

A good host would be attentive to its guest's needs. The guest will evaluate the host's response time for every request made depending on its urgency to satisfy and/or difficulty to provide an accurate answer. The dataset shows that 21% have very reactive hosts who replied within the hour, 50% of hosts answered within a few hours and 24% within the day, the remaning 2.8% refer to hosts that for some reason delivered or not an answer outside of the usual delays. 

![**Figure EDA Host Qualities**](images/EDA_Host_qualities.png)

### *About the listing's specs:*

ROOM TYPE

It can be safe to assume that the best listings will offer entire homes or appartments at the guests disposition, in Paris 88% of the listings are full homes or appartments (the latter might be a more accurate guess), then 12% offer private rooms and only 0.7% shared rooms. 

BED TYPE

The best listings will have a real bed. That's a fact!

![**Figure Room&Bed Types**](images/EDA_room_n_bed_types.png)

BEDS & BATHROOMS

If there are no bathrooms within the room or appartment, that can be inconvenient. 

The number of bathrooms range from 0 to 8 where 88% of the listings have 1 bathroom and 55% have 1 bed. On the other hand there are 33 listings that have no bathroom since they are not part of the common types of room. It turns out they are seminar rooms, showrooms, studios and "chambre de bonne" which are also studios located on the last floor and where the WC is outside somewhere along the corridor. 

There are 34 listings that dont list a bed at all. They must've forgot as they are all entire homes/apts with 0 real beds who can accommodate 2 to 7 guests and 1 to 4 extra people. So there is no such thing as a listing without beds, unless it's a seminary or showroom. Since I'm looking for factors that can affect the quality of the listing, the number of beds doesnt matter. 

On the other hand there are 7 Boutique hotel listings that report 50 bathrooms and 4 of them report 50 beds!    

AMENITIES
    
When it comes to amenities, listings will have the essential ones plus some extra that adds color and value to the listing. The **best listings** have at least the essential amenities and that's why Airbnb provides and option called "Essentials" and 94% listings did checked it. There are 136 unique amenities that I've simply categorized in 10 groups. 

![**Figure Amenities**](images/EDA_amenities_presence.png)

The figure above shows the normalized presence of each category (of amenities) in the dataset (in blue) and the size of each category (in yellow). It can be interpreted that many listings have more of electronic amenities or entertainment related than amenities related to safety and security or accessibility. 

From the graph above one would intuitively say that as long as there are amenities refering to electronics and entertainment, the listing is showing promising. 

MINIMUM AND MAXIMUM NIGHTS

These are pieces of data that will help defining the occupancy rate. There appears to be 11 outliers in the minimum_nights with a values greater than 500. On the maximum nights side there are 22618 listings with values higher than 365. 

Cutting off the outliers the distribution of the minimum number of nights is of the form: 

![**Figure Price_range**](images/EDA_minimum_nights_range.png)

With an average of 4.5 nights, I will use this as a standard value to get the occupancy rate. 

PRICE and LOCATION

The **best listing** should have a reasonable linear price/quality ratio. The less amenities a listing has (no real beds etc,) should yield a lower price and viceversa. Another intervening factor is the location. It is known that in most European cities the further you move away from the center the cheaper everything becomes. In Paris, this can be true except for certain districts like the 16th which is known to be quite expensive. In this project I will study this trinomial relationship. 

The price range resulted a bit sparse so I assigned them into broader intervals: 

![**Figure Price_range**](images/EDA_price_range.png)

The dataset shows 50 unique zipcodes and 50 unique cities (or districts) where Paris has approximately 20 zipcodes (or the number of its districts a.k.a arrondissements) and the remaining zipcodes greater than 75020 correspond to the suburbs. There are 269 listings in the suburbs, a number so negligeable it will not affect the results if removed from the dataset. In the figure below are the listings in Paris only, where there are a few outliers quite far from the city limits. The size of each listing is proportional to its price

![**Figure Districts of Paris**](images/EDA_Arrondissements_copie.png)

### *From the review section* 

NUMBER_OF_REVIEWS & REVIEWS_PER_MONTH

These descriptors will help in defining the **occupancy rate** which is the key factor of a good listing. The more reviews, the higher the occupancy rate, the higher the occupancy rate, the better the listing. 
    
Listings have 22 total reviews in average, this is considering old and new listings. A definite outlier has a bit over 600 reviews!! 
As for the reviews_per_month column there is listing #11034796 which is the only one having more than 50 reviews per month (224 to be exact) so without considering it the average reviews per month is 1. 

The dataset also provides a reviews csv file containing 1074759 reviews with its listing_id, date, reviewer_id and the review itself. I will use this table to confirm the accuracy of the two columns. 

REVIEW_SCORES..

Set of variables with scores for 6 different aspects of the listing: 

1. Rating: Variable with unknown origin, ranging from a score of 20 to 100 with a mean of 93
2. Accuracy: Refers to the accuracy of the description
3. Cleanliness: Speaks for itself. 
4. Checkin: Marks how smooth the host experienced its arrival and departure of the listing. 
5. Communication: Refers to the reactivity of the host in accurately fulfilling its host's requests. Should be correlated with host_response_time
6. Location: Refers to the quality of the neighbourhood. 
7. Value: Could be interpreted as the fairness in the price/listing ratio. 

![**Figure Districts of Paris**](images/EDA_review_scores.png)



---
# Research and Findings?? 

## Getting the Occupancy rate:

Airbnb guests may leave one review after their stay, therefore it can be used as an indicator of airbnb activity. However this option is not obligatory and therefore not all guests leave a review, so the actual booking activity could be much higher. Assuming that this practice of leaving reviews is constant.  The occupancy rate according to Airbnb would be: 

>***Occupancy_rate = MAX(average_length_of_stay, minimum_nights)x(number_of_reviews)***

An average length of stay is configured for each city, multiplied by the estimated bookings for each listing over a period gives the occupancy rate. In this dataset, the average_length_of_stay is set to 4.5 nights. 

Some things to consider is that a listing that just started hosting with Airbnb is less likely to have as much reviews as an older listing. There's at least 7000 listings whose first review dates from this year! To counteract the "old vs new" bias, I will normalize the occupancy rate per month. 

A listing with high availability is susceptible to get more reviews and thus a higher occupancy rate. An Airbnb host can setup a calendar for their listing so that it's only available for a few days or weeks a year.  Other listings are available all year round (except for when it is already booked). 

This poses an ambiguity problem since it can either mean the number of days the listing is supposed to be available all year roudn but it can also mean the remaining days a listing is available for booking thus having a low availability. 

In order to get an accurate rate, I will not consider the availability. So the formula shorts to: 

> ***Average occupancy rate per month = minimum nights x average reviews per month***

..which basically says that the higher the rate of bookings (reviews) per month, the higher the occupancy rate and thus the minimum nights can be removed as it's just a constant. The distribution yielded not a very wide spreaded shape. 

Most listings have a really low rate compared to few of their fellow outlier listings, the next steps are an attempt to extend the spectrum of normal rate listings and compress that of the outliers. 

An attempt to exploit the discriminating power of this rate is to ceiling those outliers to the q3+1.5*IQR value.

![**Figure Districts of Paris**](images/PROJ_Occupancy_rate_map.png)


---
## Getting the Listings review rate:

#### How can a grading system be implemented using the following parameters: 
- HOST_SUPERHOST: will have values 100 or 0
- ROOM_TYPE: will be mapped with normalized values as such: 
'Entire home/apt' 100 pts, 'Private room' 50 pts and 'Shared room' 0 pts
- BED_TYPE: will be mapped with normalized values as such: 
Real bed: 100 pts, Pull-out Sofa, Couch, Futon and Airbed: 50pts
- BATHROOMS: will be normalized by 100 as the more bathrooms the fancier the listing is. 
- AMENITIES: has 100 for any listing with 100% of the essential amenities.   
- REVIEW SCORES: They are already normalized scores. 

![**Figure Districts of Paris**](images/PROJ_listing_review_rate.png)




___
# Further Research and Analysis

In order to get the most popular listings, the scenario #2 looks more promising as the occupancy rate shows a higher discriminatory power. The next steps are to determine the best listing according to the host qualities, listing specs, location (in terms of price and public transportation) and review scores. Thirdly I will insert the coordinates of the touristy places and get those listings that are closest and pull out the proportion of popular vs unpopular listings. 

There is room to implement association rules in variables such as amenities and host_verifications. As a host the re is interest in what sort of amenities are more apealing to guests. Which amenities can be considered luxurious and which are useless? 