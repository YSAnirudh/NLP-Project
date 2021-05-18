import pandas as pd
import wiki_scrape
from time import time
from tqdm import tqdm
import entity_pair_ext
import coref_train
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
    # text = pd.read_csv('wiki_text.csv').to_numpy().tolist()
    # print(len(text))
    # timeS = time()
    # knowledge_graph = KnowledgeGraph(parallel=True)
    # for i in range(len(text)):
    #     knowledge_graph.add_text(text[i])
    # knowledge_graph.build_knowledge_graph()
    # print(knowledge_graph.entity_pairs_df)
    # knowledge_graph.entity_pairs_df.to_csv("knowledge_graph.csv", index = False)
    # timeE = time()
    # print("Time:", timeE - timeS)
    wiki_scraper = WikiScrape("https://en.wikipedia.org/wiki/Cristiano_Ronaldo", index = True, parallel=True)
    wiki_scraper.scrape_all_pages()


# data_into_neo4j("stanopenie.csv")
