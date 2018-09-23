from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from bisect import bisect

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

'''---------------------PARIS MAP PLOT---------------------'''


def print_listings(m, s):
    fig = plt.figure(figsize=(25,10))
    if 'color' in m.columns:
        c = m.color.values
    else:
        c = 'royalblue'
    plt.scatter(m.longitude.values, m.latitude.values, s=s, c=c, zorder=2, alpha=1)
    plt.show()

    
'''Function that contains latitude and longitudes of the districts limits'''
def get_arrondissement_info():
# Arrondissements Coordinates
    arrn = {                                 # OUTER
        'a16_17' : [48.878821, 2.279310],          # Outside perif
        'a15_16': [48.839349, 2.268270],           # Mid river on perif
        'a14_15' : [48.825465, 2.301253],
        'a13_14' : [48.816552, 2.344029],
        'a12_13' : [48.826531, 2.388806],
        'a12_20' : [48.846605, 2.416004],
        'a19_20' : [48.878434, 2.410734],
        'a18_19' : [48.901810, 2.370136],
        'a17_18' : [48.900997, 2.330128],
        'a4_5_12_13' : [48.846113, 2.364457],  # INNER
        'a5_6_13_14' : [48.839663, 2.336769],
        'a6_7_15' : [48.846851, 2.316689],
        'a6_14_15' : [48.843632, 2.324635],
        'a7_15' : [48.857565, 2.290917],
        'a8_9_17_18' : [48.883555, 2.327461],
        'a8_16_17' : [48.873700, 2.294935],
        'a8_16' : [48.863475, 2.301591],
        'a9_10_18' : [48.883696, 2.349539],
        'a10_11_19_20' : [48.872101, 2.377034],
        'a10_18_19' : [48.884375, 2.362951],
        'a11_12_20' : [48.848305, 2.395834]

    }

    # Mark the Arrondissements.
    plabel = []
    plat = []
    plon = []
    for e in arrn:
        plabel.append(e)
        plat.append(arrn[e][0])
        plon.append(arrn[e][1])

    colors_per_zip_limit = ['black','blue','cyan','darkslateblue','gold',
'green','grey','lime','magenta','maroon',
'navy','olive','orange','purple',
'pink','red','royalblue','silver','teal',
'violet','yellow']


    return plat, plon, plabel, colors_per_zip_limit

'''Function that scatter plots whatever coordinates it's given
coord is a dataframe with columns:
latitude, longitude, zipcode and a 4th column
res - resolution of the basemap [c for crude, h for high]
arrn_lim - If True then plot the district limits
title_map - Title of plot
l_size - size of listings if False then it plots listings per the 4th column
l_color - color of listings if False then it plots listings colored per zipcode
l_alpha - alpha value of listings.
'''

def print_map(coord, res='c', arrn_lim=1, title_map='Paris', l_size=10, l_color='blue', l_alf=0.4):

    fig, ax = plt.subplots(figsize=(25,10)) #12.5,5)) # 25/10

    m = Basemap(projection='cyl',
                resolution=res,
                llcrnrlon=2.25, llcrnrlat=48.813500,
                urcrnrlon=2.421, urcrnrlat=48.906000)
    m.drawrivers(color='lightblue', linewidth=2)

    # Insert Paris background image
    img = mpimg.imread('../images/Paris_map.jpg')
    im = m.imshow(img, extent=(2.25,2.45,48.797622,48.925604), alpha=0.4, zorder=1, origin='upper')

    # Arrondissements limits
    if arrn_lim:
        plat, plon, plabel, colors_per_zip_limit = get_arrondissement_info()
        arr_xpt, arr_ypt = m(plon, plat)
        m.scatter(arr_xpt, arr_ypt, s=100, c=colors_per_zip_limit, zorder=3, marker='X')

    # Insert Listings

    if not l_size:
        breakpoints = [100, 200, 1000, 10000]
        coord['price_cat'] = coord.price.apply(get_price_interval)
        coord['price_size'] = coord.price_cat.apply(lambda x: (breakpoints.index(x)*15))
        l_size = coord.price_size.values
    if not l_color:
        l_color = coord['zipcode'].apply(lambda x: cpzl[x])

    lis_xpt, lis_ypt = m(coord.longitude.values, coord.latitude.values)
    m.scatter(lis_xpt, lis_ypt, s=l_size, c=l_color, zorder=2, alpha=l_alf)


    plt.title(title_map)
    plt.show()
    fig.savefig('../images/EDA_Arrondissements.png', bbox_inches='tight')

def get_price_interval(total):
    breakpoints = [100, 200, 1000, 10000]
    return breakpoints[bisect(breakpoints, total)-1]

def print_summary(l,idx,s_col):

    print(l.loc[idx,'name'])
    print(l.loc[idx,'summary'],'\n')
    print(l.loc[idx,'space'],'\n')
    print(l.loc[idx,'accommodates'],'people\n')
    print(l.loc[idx,'description'],'\n')
    print(l.loc[idx,'neighborhood_overview'],'\n')
    print(l.loc[idx,'transit'],'\n')
    print(l.loc[idx,'access'],'\n')
    print(l.loc[idx,'notes'],'\n')
    print(l.loc[idx,'interaction'],'\n')
    print(l.loc[idx,s_col])

def print_paris(centr, abrnt, res, title_map, l_size, l_alf, im_alf):
    centroid = centr
    c_color = centroid.color.values
    l_color = abrnt.color.values

    fig, ax = plt.subplots(figsize=(25,10)) #12.5,5)) # 25/10

    m = Basemap(projection='cyl',
                resolution=res,
                llcrnrlon=2.25, llcrnrlat=48.813500,
                urcrnrlon=2.421, urcrnrlat=48.906000)
    m.drawrivers(color='lightblue', linewidth=2)

    # Insert Paris background image
    img = mpimg.imread('../images/Paris_map.jpg')
    im = m.imshow(img, extent=(2.25,2.45,48.797622,48.925604), alpha=im_alf, zorder=1, origin='upper')

    if l_size is False:
        l_size = 200

    # Plot Centroids
    cis_xpt, cis_ypt = m(centroid.longitude.values, centroid.latitude.values)
    m.scatter(cis_xpt, cis_ypt, s=l_size, c=c_color, zorder=2, marker='X')

    # Plot outliers.
    outx, outy = m(abrnt.longitude.values, abrnt.latitude.values)
    m.scatter(outx, outy, s=50, c=l_color, zorder=3, marker='D', alpha=0.7)

    plt.title(title_map)
    plt.show()

    fig.savefig('../images/'+title_map+'.png', bbox_inches='tight')

#----------------------------- VARS---------------------

cpzl = {
75001:'yellow',75002:'blue',75003:'dimgray',75004:'darkslateblue',75005:'olive',
75006:'green',75007:'cyan',75008:'lime',75009:'purple',75010:'maroon',
75011:'cyan',75012:'gold',75013:'red',75014:'purple',75015:'royalblue',
75016:'orange',75017:'magenta',75018:'teal',75019:'navy',75020:'violet',
92100:'black', 92110:'black', 92120:'black', 92130:'black', 92150:'black',
92170:'black', 92200:'black', 92240:'black', 92300:'black', 92310:'black',
92600:'black', 93100:'black', 93170:'black', 93260:'black', 93300:'black',
93310:'black', 93400:'black', 93500:'black', 94120:'black', 94130:'black',
94160:'black', 94200:'black', 94220:'black', 94250:'black', 94270:'black',
94300:'black', 94410:'black', 94700:'black', 94800:'black', 95170:'black'
}
