from matplotlib.pyplot import text
import spacy
import neuralcoref
import re
import pandas as pd
from tqdm import tqdm
import concurrent

class KnowledgeGraph():
    def __init__(self, text = "",coref:bool=True, lemmatize:bool = True, parallel:bool = False):
        self.texts = []
        self.coref = coref
        self.sentences = []
        self.lemmatize = lemmatize
        self.nlp = spacy.load('en_core_web_sm')
        self.parallel = parallel
        neuralcoref.add_to_pipe(self.nlp)
        if (text != ""):
            self.add_text(text)

        self.entity_pairs_df = pd.DataFrame()
    
    def parallel_extract(self, textInd):
        print("Text No. " + str(textInd+1))
        print(len(self.texts[textInd]))
        #print("Splitting into sentences for Text No. " + str(textInd+1))
        # print(type(self.texts[textInd][0]))
        # print(self.texts[textInd])
        sentences = self.coref_resolution(self.texts[textInd])
        #print("Splitting into Sentences Done.")
        #print("Extracting Entities for Text No. " + str(textInd+1))
        return self.get_entity_pairs(sentences=sentences)

    def build_knowledge_graph(self):
        if (not self.parallel):
            text_data_frames = [None for i in range(len(self.texts))]
            for textInd in tqdm(range(len(self.texts))):
                sentences = []
                # f = open("sentences.txt", 'r')
                # sentences = f.read().split("\n")
                # print("Splitting into sentences for Text No. " + str(textInd+1))
                sentences = self.coref_resolution(self.texts[textInd])
                # print("Splitting into Sentences Done.")
                # print("Extracting Entities for Text No. " + str(textInd+1))
                text_data_frames[textInd] = self.get_entity_pairs(sentences=sentences)
            self.entity_pairs_df = pd.concat(text_data_frames)
        else:
            text_data_frames = [None for i in range(len(self.texts))]
            intervals = []
            # print(len(self.texts))
            for i in range(len(self.texts)):
                intervals.append(i)

            # print(intervals)
            with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
                for (i, data_frame) in tqdm(zip(intervals, executor.map(self.parallel_extract, intervals))):
                    text_data_frames[i] = data_frame
            self.entity_pairs_df = pd.concat(text_data_frames)

    def add_text(self, text):
        if (text != ""):
            self.texts.append(text)

    def no_of_texts(self):
        print("You have " + str(len(self.texts)) + " Texts")


    def coref_resolution(self, text):
        # print("\tSplitting into Sentences tagged by SpaCy...")
        text = re.sub(r'\n+', '.', text)  # replace multiple newlines with period
        text = re.sub(r'\[\d+\]', ' ', text)  # remove reference numbers
        text = self.nlp(text)
        if (self.coref):
            # print("\tResolving Coreference...")
            texty = text._.coref_resolved
            text = self.nlp(texty)
        # else:
            # print("\tNo Coreference Resolution...")
        sentences = [sent.string.strip() for sent in text.sents]
        return sentences
    
    def entity_extraction_by_ROOT(self, sent):
        ent_pairs = []
        for token in sent:
            try:
                if token.dep_ not in ('obj', 'dobj'):  # identify object nodes
                    continue
                subject = [w for w in token.head.lefts if w.dep_
                        in ('subj', 'nsubj')]  # identify subject nodes
                if subject:
                    subject = subject[0]
                    # identify relationship by root dependency
                    relation = [w for w in token.ancestors if w.dep_ == 'ROOT']
                    if relation:
                        relation = relation[0]
                        # add adposition or particle to relationship

                        if relation.nbor(1).pos_ in ('ADP', 'PART'):
                            relation = ' '.join((str(relation), str(relation.nbor(1))))
                    else:
                        relation = 'unknown'

                    tok = [subject, relation, token]
                    ent = [subject, relation.lemma_, token]
                    entPlusTok = self.get_full_entity_pair(ent, tok)
                    if (entPlusTok):
                        ent_pairs.append(entPlusTok)
            except Exception:
                pass
    
    def entity_extraction_by_SUBJECT(self, sent):
        ent = ["" for i in range(3)]
        tok = [None for i in range(3)]
        text = ""
        ent_pairs = []
        for token in sent:
            if (ent[0] == "" and token.dep_ == "nsubj" or token.dep_ == "subj"):
                ent[0] = token.text
                tok[0] = token
            elif (ent[0] != ""and ent[1] == ""):
                if (token.pos_ == "VERB"):
                    try:
                        if token.nbor(1).pos_ in ('ADP', 'PART'):
                            text = ' '.join((str(token.lemma_), str(token.nbor(1).lemma_)))
                    except:
                        pass
                    if (text == ""):
                        ent[1] = token.lemma_
                    else:
                        ent[1] = text
                    tok[1] = token
                if (token.pos_ == "ADP"):
                    try:
                        if token.nbor(1).pos_ in ('VERB'):
                            text = ' '.join((str(token.lemma_), str(token.nbor(1).lemma_)))
                    except:
                        pass
                    if (text == ""):
                        ent[1] = token.lemma_
                    else:
                        ent[1] = text
                    tok[1] = token
            elif (ent[0] != "" and ent[1] != "" and ent[2] == ""):
                if (token.dep_ == 'nsubj' or token.dep_ == 'subj' or token.dep_ == 'obj' or token.dep_ == 'dobj' or token.dep_ == "pobj"):
                    ent[2] = token.text
                    tok[2] = token
                    entPlusTok = self.get_full_entity_pair(ent, tok)
                    if (entPlusTok):
                        ent_pairs.append(entPlusTok)
                    tok[1] = None
                    ent[1] = ""
                    tok[2] = None
                    ent[2] = ""
        return ent_pairs
    
    def entity_extraction_by_SENTENCE(self, sent):
        ent = ["" for i in range(3)]
        tok = [None for i in range(3)]
        text = ""
        ent_pairs = []
        for token in sent:
            if (ent[0] == ""):
                if ((token.pos_ == "NOUN" or token.pos_ == "PROPN")
                    and (token.dep_ == 'subj' or token.dep_ == 'nsubj' or token.dep_ == 'obj' or token.dep_ == 'dobj')):
                    tok[0] = token
                    ent[0] = token.text
            
            elif (ent[0] != "" and ent[1] == ""):
                if ((token.pos_ == "NOUN" or token.pos_ == "PROPN")
                and (token.dep_ == 'subj' or token.dep_ == 'nsubj' or token.dep_ == 'obj' or token.dep_ == 'dobj')):
                    tok[0] = token
                    ent[0] = token.text 
                if (token.pos_ == "VERB" and (tok[0].pos_ == "NOUN" or tok[0].pos_ == "PROPN")):
                    try: 
                        if token.nbor(1).pos_ in ('ADP', 'PART'):
                            text = ' '.join((str(token.lemma_), str(token.nbor(1).lemma_)))
                    except:
                        pass
                    tok[1] = token
                    if (text == ""):
                        ent[1] = token.lemma_
                    else:
                        ent[1] = text
            
            elif (ent[0] != "" and ent[1] != "" and ent[2] == ""):
                if ((token.pos_ == "NOUN" or token.pos_ == "PROPN") and (tok[1].pos_ == "VERB")
                    and (tok[0].pos_ == "NOUN" or tok[0].pos_ == "PROPN")):
                    tok[2] = token
                    ent[2] = token.text
                if ((token.pos_ == "NOUN" or token.pos_ == "PROPN" 
                or token.dep_ == 'subj' or token.dep_ == 'nsubj' or token.dep_ == 'obj' 
                or token.dep_ == 'dobj' or token.dep_ == 'pobj') and (tok[1].pos_ == "ADP")
                    and (tok[0].pos_ == "NOUN" or tok[0].pos_ == "PROPN")):
                    tok[2] = token
                    ent[2] = token.text
            
            elif (ent[0] != "" and ent[1] != "" and ent[2] != ""):
                if (token.pos_ == "PUNCT" and (token.text == ";" or token.text == ".")):
                    ent = self.clear_entity_pair_value(ent, tok)
                if (token.pos_ == "VERB"):
                    try:
                        if token.nbor(1).pos_ in ('ADP', 'PART'):
                            text = ' '.join((str(token.lemma_), str(token.nbor(1).lemma_)))
                    except:
                        pass
                    tok[0] = tok[2]
                    ent[0] = ent[2]
                    tok[1] = token
                    if (text == ""):
                        ent[1] = token.lemma_
                    else:
                        ent[1] = text
                    tok[2] = None
                    ent[2] = ""
                if (token.pos_ == "ADP"):
                    try:
                        if token.nbor(1).pos_ in ('VERB'):
                            text = ' '.join((str(token.lemma_), str(token.nbor(1).lemma_)))
                    except:
                        pass
                    tok[0] = tok[2]
                    ent[0] = ent[2]
                    tok[1] = token
                    if (text == ""):
                        ent[1] = token.lemma_
                    else:
                        ent[1] = text
                    tok[2] = None
                    ent[2] = ""
                if ((token.pos_ == "PUNCT" and token.text == ",") 
                    or (token.pos_ == "CCONJ" and (token.text == "and" or token.text == "or"))):
                    ent[2] = ""
                    tok[2] = None
            entPlusTok = self.get_full_entity_pair(ent, tok)
            if (entPlusTok):
                ent_pairs.append(entPlusTok)
        return ent_pairs

    def refine_entity(self, ent, sentLen, lemmatize):
        unwanted_tokens = (
            'PRON',  # pronouns
            'PART',  # particle
            'DET',  # determiner
            'SCONJ',  # subordinating conjunction
            'PUNCT',  # punctuation
            'SYM',  # symbol
            'X',  # other
        )
        unwanted = ('DET')
        if (lemmatize):
            ent_type = ent.ent_type_  # get entity type
            if ent_type == '':
                ent_type = 'NOUN_CHUNK'
                ent = ' '.join(str(t.text) for t in
                                self.nlp(str(ent.lemma_)) if t.pos_
                                not in unwanted_tokens and t.is_stop == False)
            elif ent_type in ('NOMINAL', 'CARDINAL', 'ORDINAL') and str(ent.lemma_).find(' ') == -1:
                refined = ''
                for i in range(sentLen - ent.i):
                    if ent.nbor(i).pos_ not in ('VERB', 'PUNCT'):
                        refined += ' ' + str(ent.nbor(i).lemma_)
                    else:
                        ent = refined.strip()
                        break
            else :
                ent = ' '.join(str(t.text) for t in
                                self.nlp(str(ent.lemma_)) if t.pos_
                                not in unwanted and t.is_stop == False)
            return ent, ent_type
        else:
            ent_type = ent.ent_type_  # get entity type
            if ent_type == '':
                ent_type = 'NOUN_CHUNK'
                ent = ' '.join(str(t.text) for t in
                                self.nlp(str(ent)) if t.pos_
                                not in unwanted_tokens and t.is_stop == False)
            elif ent_type in ('NOMINAL', 'CARDINAL', 'ORDINAL') and str(ent).find(' ') == -1:
                refined = ''
                for i in range(sentLen - ent.i):
                    if ent.nbor(i).pos_ not in ('VERB', 'PUNCT'):
                        refined += ' ' + str(ent.nbor(i))
                    else:
                        ent = refined.strip()
                        break
            else :
                ent = ' '.join(str(t.text) for t in
                                self.nlp(str(ent)) if t.pos_
                                not in unwanted and t.is_stop == False)
            return ent, ent_type
    
    def get_entity_pairs(self, sentences):
        non_refined_entity_pairs = self.entity_pair_extraction(sentences)
        refined_entity_pairs = []
        # print("\tRefining Entities: ")
        for ent_ind in range(len(non_refined_entity_pairs)):
            subject, subject_type = self.refine_entity(
                non_refined_entity_pairs[ent_ind][0], 
                non_refined_entity_pairs[ent_ind][6], 
                self.lemmatize)
            object, object_type = self.refine_entity(
                non_refined_entity_pairs[ent_ind][2], 
                non_refined_entity_pairs[ent_ind][6], 
                self.lemmatize)
            relation = non_refined_entity_pairs[ent_ind][4].lower()
            try:
                if (subject.lower() != "" and relation != "" and object.lower() != ""):
                    refined_entity_pairs.append(
                        [subject.lower(), 
                        relation, 
                        object.lower(), 
                        subject_type, 
                        object_type])
            except:
                pass
        return pd.DataFrame(
            refined_entity_pairs, 
            columns=['subject', 'relation', 'object', 'subject_type', 'object_type'])
    
    def entity_pair_extraction(self, sentences, index:int = -1):
        entity_token_pairs = []
        # print("\tGetting Raw Entities: ")
        for sentInd in range(len(sentences)):
            ind = sentInd
            if(index >= 0):
                ind = index
            if (sentInd == ind):
                if (sentences[sentInd] == ""):
                    continue
                if (sentences[sentInd][0] == '.'):
                    sentences[sentInd] = sentences[sentInd][1:]
                sentences[sentInd] = self.nlp(sentences[sentInd])
                spans = list(sentences[sentInd].ents) + list(sentences[sentInd].noun_chunks)  # collect nodes
                spans = spacy.util.filter_spans(spans)
                with sentences[sentInd].retokenize() as retokenizer:
                    [retokenizer.merge(span, attrs={'tag': span.root.tag,
                                                    'dep': span.root.dep}) for span in spans]
                # deps = [token.dep_ for token in sentences[sentInd]]
                # tokens = [token for token in sentences[sentInd]]
                # pos = [token.pos_ for token in sentences[sentInd]]
                # print(deps)
                # print(tokens)
                # print(pos)
                # print(sentences[sentInd])
                all_pairs = []
                # pairs1 = self.entity_extraction_by_SUBJECT(sentences[sentInd])
                # if (pairs1):
                #     all_pairs = all_pairs + pairs1
                pairs2 = self.entity_extraction_by_SENTENCE(sentences[sentInd])
                if (pairs2):
                    all_pairs = all_pairs + pairs2
                pairs3 = self.entity_extraction_by_ROOT(sentences[sentInd])
                if (pairs3):
                    all_pairs = all_pairs + pairs3

                tok_df = pd.DataFrame(columns=['subject', 'relation', 'object', 'entsub', 'entrel', 'entobj', 'sentlen'])
                
                for j in range(len(all_pairs)):
                    abcd = all_pairs[j][1] + all_pairs[j][0]
                    abcd.append(len(sentences[sentInd]))
                    tok_df.loc[j] = abcd
                non_dup_entry = tok_df.drop_duplicates().to_numpy().tolist()
                entity_token_pairs = entity_token_pairs + non_dup_entry
        return entity_token_pairs

    def clear_entity_pair_value(self, mrEnt, tok):
        if (mrEnt[0] != "" and mrEnt[1] != "" and mrEnt[2] != ""):
            mrEnt[0] = ""
            mrEnt[1] = ""
            mrEnt[2] = ""
            tok[0] = None
            tok[1] = None
            tok[2] = None
        return mrEnt
    
    def get_full_entity_pair(self, mrEnt, tok):
        ent = []
        token = []
        if (mrEnt[0] != "" and mrEnt[1] != "" and mrEnt[2] != ""):
            ent[:] = mrEnt[:]
            token[:] = tok[:]
            return [ent, token]