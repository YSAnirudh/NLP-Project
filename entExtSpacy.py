import pandas as pd
import wiki_scrape
from time import time
from tqdm import tqdm
from py2neo import Graph, Node, Relationship
from entity_pair_extraction import KnowledgeGraph
from wiki_scrape import WikiScrape

# def data_into_neo4j(filename):
#     graph = Graph("bolt://localhost:7687", auth=("neo4j", "Ysani123"))
#     df = pd.read_csv(filename)
#     for index, row in df.iterrows():
#         tx = graph.begin()
#         a = Node('Subject', name = row['subject'])
#         tx.create(a)
#         b = Node('Object', name = row['object'])
#         tx.create(b)
#         ab = Relationship(a, row['relation'], b)
#         tx.create(ab)
#         tx.commit()

# text = wiki_scrape.scrapeWikiArticle("https://en.wikipedia.org/wiki/Cristiano_Ronaldo", index= False)
# # df = triples(text=text[0][0])

# # df.to_csv("openie.csv", index = False)
# x = time()

# #resDataFrames, tCorefS, tCorefE, lengthEnt = entity_pair_ext.get_entity_pairs(text[0][0])
# #resDataFrames.to_csv("spacy.csv", index=False)

# sentences = coref_train.coref_resolution(text[0][0])
# print(len(sentences))
# y = time()
# print("Spacy Time:", y-x)
# f = open("sentences.txt", 'w')
# for sent in sentences:
#     try:
#         f.write(sent + "\n")
#     except UnicodeEncodeError as e:
#         print(e)
#         pass
# f.close()

# f = open("sentences.txt", 'r')

# sentences = f.read().split("\n")
#print(sentences)
if __name__ == '__main__':
    text_df = pd.read_csv('wiki_text.csv')
    text = text_df['Wiki'].to_numpy().tolist()
    print(len(text))
    # print(type(text[0]))
    # print(text)
    # print(text[0])
    timeS = time()
    # print(text[2][1][0])
    # print(text[0])
    knowledge_graph = KnowledgeGraph(parallel=True)
    for i in range(len(text)):
        knowledge_graph.add_text(text[i])
    knowledge_graph.build_knowledge_graph()
    print(knowledge_graph.entity_pairs_df)
    knowledge_graph.entity_pairs_df.to_csv("knowledge_graph.csv", index = False)
    timeE = time()
    print("Time:", timeE - timeS)



    # wiki_scraper = WikiScrape("https://en.wikipedia.org/wiki/Elon_Musk", index = True, parallel=True)
    # wiki_scraper.add_url("https://en.wikipedia.org/wiki/Miss_Universe", index=True)
    # wiki_scraper.add_url("https://en.wikipedia.org/wiki/Academy_Awards", index=True)
    # wiki_scraper.add_url("https://en.wikipedia.org/wiki/Karim_Benzema", index=True)
    # wiki_scraper.add_url("https://en.wikipedia.org/wiki/Gaza%E2%80%93Israel_conflict", index=True)
    # wiki_scraper.scrape_all_pages()
    # tweets_df2 = pd.DataFrame(wiki_scraper.wiki_pages, columns=['Wiki'])
    # tweets_df2.to_csv('wiki_text.csv',encoding="utf-8")

# data_into_neo4j("stanopenie.csv")
