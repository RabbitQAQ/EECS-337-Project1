# This package will contain the spiders of your Scrapy project
#
# Please refer to the documentation for information on how to create and manage
# your spiders.
from spiders.ggspider import year

# clean file
f = open('../data/AwardCategories' + year + '.txt', 'w+')
f.write('')
f.close()