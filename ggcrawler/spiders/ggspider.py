from scrapy import Spider

from ggcrawler.items import CategoryItem

year = "2015"

class GlodenGlobesSpider(Spider):
    name = 'gloden_globes'
    start_urls = ["https://www.goldenglobes.com/winners-nominees/" + year]

    def parse(self, response):
        item = CategoryItem()
        categories = response.xpath('//a[contains(@href, "all#category")]')
        for category in categories:
            categoryName = category.xpath('./text()').extract()[0]
            item['category'] = categoryName
            yield item

