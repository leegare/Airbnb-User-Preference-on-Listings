"""NOT USED ANYMORE"""

orate = {0.4:5, 0.6:15, 0.8:20, 0.9:25, 0.925:35, 0.95:45, 0.975:55, 1:75}

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

"""NOT USED ANYMORE"""
'''Function that reduces the number of unique values into larger intervals'''
def get_interval_on_rate(total):
    return orate[bisect(orate, total)-1]

    from bisect import bisect
    orate_dico = {0:5, 0.2:10, 0.4:15, 0.8:20, 0.9:25, 0.925:30, 0.95:35, 0.975:40, 1:75}
    oratae = [0, 0.2, 0.4, 0.8, 0.9, 0.925, 0.95, 0.975, 1]


    def get_interval_on_rate(o_rate):
        main_intrvl = orate[bisect(orate, o_rate)-1]
        return orate_dico[main_intrvl]

    o_rate = data.loc[(data.zipcode<76000)&(data.occupancy_rate<1),['longitude','latitude','zipcode','occupancy_rate']]
    #o_rate ['interval_o_rate'] = o_rate.occupancy_rate.apply(get_interval_on_rate)
