from functools import partial
import wikipediaapi
import concurrent.futures
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import concurrent
from itertools import repeat

class WikiScrape():
    def __init__(self, url = "", index=False, max_pages=-1, parallel:bool = False):
        self.urls = []
        self.wiki_pages = []
        self.parallel = parallel
        if (url != ""):
            self.add_url(url, index, max_pages)
        self.hrefs = set()
    def add_url(self, url, index = False, max_pages = -1):
        if (url != ""):
            self.urls.append([url, index, max_pages])

    def scrape_all_pages(self):
        texts = []
        for i in range(len(self.urls)):
            scraped = []
            if (not self.parallel):
                scraped, leng = self.scrape_wiki_pages(self.urls[i][0], self.urls[i][1], self.urls[i][2])
            else:
                scraped, leng = self.parallel_scrape_wiki_pages(self.urls[i][0], self.urls[i][1], self.urls[i][2])
            for i in range(len(scraped)):
                texts.append(scraped[i])
                # print(len(scraped[i]))
                if (scraped[i][1]):
                    print("The Main Page:", scraped[i][0])
                
        print(len(texts))
        self.wiki_pages = texts
    
    def parallel_scrape_helper(self, url):
        response = requests.get(
            url=url,
        )
        # print(url)
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
        except requests.exceptions.ConnectionError as e:
            print("Continuing due to Connection Error.")
            return
        
        page_text = []
        text = ""
        for para in soup.find(id="bodyContent").find_all("p"):
            text += para.text
        page_text.append(text)

        return page_text, len(page_text)
    
    def parallel_scrape_wiki_pages(self, url, index = False, max_pages = -1):
        response = requests.get(
            url=url,
        )
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
        except requests.exceptions.ConnectionError as e:
            print("Continuing due to Connection Error.")
            return
        
        page_text = []
        text = ""
        for para in soup.find(id="bodyContent").find_all("p"):
            text += para.text
        page_text.append([text, True])

        allLinks = soup.find(id="bodyContent").find_all("a")
        if (index == True):
            i = 0
            intervals = []
            for ind in range(max_pages if max_pages != -1 else len(allLinks)):
                href = allLinks[ind].get('href')
                if (not href):
                    continue
                if allLinks[ind]['href'].find("/wiki/") == -1:
                    continue
                if allLinks[ind]['href'][0] != '/':
                    if (allLinks[ind]['href'] not in self.hrefs):
                        intervals.append(allLinks[ind]['href'])
                        self.hrefs.add(allLinks[ind]['href'])
                if (href not in self.hrefs):
                    intervals.append("https://en.wikipedia.org/" + allLinks[ind]['href'])
                    self.hrefs.add(allLinks[ind]['href'])
                    continue
            print("Total No of Links for", url, ":", len(intervals))
            with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
                for page_data in tqdm(executor.map(self.parallel_scrape_helper, intervals)):
                    if (page_data[0] and page_data[1]):
                        page_text.append([page_data[0], False])
        return page_text, len(page_text)

    def scrape_wiki_pages(self, url, index = False, max_pages = -1):
        response = requests.get(
            url=url,
        )
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
        except requests.exceptions.ConnectionError as e:
            print("Continuing due to Connection Error.")
            return
        
        page_text = []
        text = ""
        for para in soup.find(id="bodyContent").find_all("p"):
            text += para.text
        page_text.append([text, True])

        allLinks = soup.find(id="bodyContent").find_all("a")
        if (index == True):
            i = 0
            # lengthY = max_pages if max_pages != -1 else len(allLinks) 
            for ind in tqdm(range(max_pages if max_pages != -1 else len(allLinks))):
                href = allLinks[ind].get('href')
                if not href:
                    continue
                if allLinks[ind]['href'].find("/wiki/") == -1:
                    continue
                if allLinks[ind]['href'][0] != '/':
                    if (allLinks[ind]['href'] not in self.hrefs):
                        pg, length, _ = self.scrape_wiki_pages("" + allLinks[ind]['href'], index= False)
                        page_text.append([pg[0], False])
                        self.hrefs.add(allLinks[ind]['href'])
                        continue
                if (href not in self.hrefs):
                    pg, length, _ = self.scrape_wiki_pages("https://en.wikipedia.org/" + allLinks[ind]['href'], index= False)
                    page_text.append([pg[0], False])
                    self.hrefs.add(allLinks[ind]['href'])
                    continue
        return page_text, len(page_text)
        
    def wiki_page(self, page_name):
        wiki_api = wikipediaapi.Wikipedia(language='en',
                extract_format=wikipediaapi.ExtractFormat.WIKI)
        print("Scrape Start")
        page_name = wiki_api.page(page_name)
        if not page_name.exists():
            print('Page {} does not exist.'.format(page_name))
            return

        page_data = pd.DataFrame({
            'page': page_name,
            'text': page_name.text,
            'link': page_name.fullurl,
            'categories': [[y[9:] for y in
                        list(page_name.categories.keys())]],
            })
        print("Scraped", page_name)
        return page_data

    def wiki_scrape(topic):
        def wiki_link(link):
            try:
                page = wiki_api.page(link)
                if page.exists():
                    return {'page': link, 'text': page.text, 'link': page.fullurl,
                            'categories': list(page.categories.keys())}
            except:
                return None

        wiki_api = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)

        page = wiki_api.page(topic)

        if not page.exists():
            print("Page {} doesn't exist.".format(topic))

        page_links = list(page.links.keys())
        print(len(page_links))
        sources = [{'page': topic, 'text': page.text, 'link': page.fullurl,'categories': list(page.categories.keys())}]
        print("Scraping links START...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as parallel:
            future_links = {parallel.submit(wiki_link, link): link for link in page_links}
            for future in concurrent.futures.as_completed(future_links):
                data = future.result()
                sources.append(data) if data else None
        print("Scraping links DONE...")   
        sources = pd.DataFrame(sources)
        sources['categories'] = sources.categories.apply(lambda x: [y[9:] for y in x])
        sources['topic'] = topic
        print("Scraped:", len(sources), "Pages on", topic)
        return sources


# text = scrapeWikiArticle("https://en.wikipedia.org/wiki/Elon_Musk", index=True, max_pages=30)
# #print(text[0][1])
# wiki_df = pd.DataFrame(text[0])
# wiki_df.to_csv('wiki_text.csv', index=False)