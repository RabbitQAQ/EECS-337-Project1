# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://doc.scrapy.org/en/latest/topics/item-pipeline.html
from spiders.ggspider import year


class GgcrawlerPipeline(object):
    def process_item(self, item, spider):
        f = open('../AwardCategories' + year + '.txt', 'a+')
        f.write(item['category'] + '\n')
        f.close()
        return item
