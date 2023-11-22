import numpy as np

#%%
class Feature():
    "object to extract a feature set from a head and a child"
    def __init__(self, sentence, head_id, child_id):
        self.features = set()
        self.make_arc_features(sentence, head_id, child_id)
        
    def __str__(self):
        return str(self.features)
    
    def __call__(self):
        return str(self.features)
        
    def make_arc_features(self, sentence, head_id, child_id):
        """
        takes in a tuple of <head, child> and extract all features
        templatename:value:direction:distance
        For example: "The cute dog likes apples", one of the features for the arc of <likes, dog> could be hform+likes+left+1 
        
        Feature templates are taken from McDonald et al. (2005)

        """
        def add(name, *arg): 
            "add feature generated from templates"
            self.features.add("+".join((name,)+tuple(arg)))
            
        def add_bpos(sentence, head_id, child_id):
            "add pos between head and dependent"
            bpos = set()
            
            if abs(head_id - child_id) == 1:
                bpos = {"__NULL__"}
                
            elif head_id < child_id:
                for b_id in range(head_id+1, child_id):
                    bpos.add(sentence.tokens[b_id].pos) 
            else: 
                for b_id in range(child_id+1, head_id):
                    bpos.add(sentence.tokens[b_id].pos) 
                
            return bpos 
        
        head_id = int(head_id)
        child_id = int(child_id)
        
        head_token = sentence.tokens[head_id]
        child_token = sentence.tokens[child_id]
        
        distance = str(abs(head_id - child_id))
        
        if int(head_id) < int(child_id):
            direction = "l"
        else:
            direction = "r"

        add("01", head_token.lemma, direction, distance) #hform
        add("02", child_token.lemma, direction, distance) #dform
        add("03", head_token.pos, direction, distance) #hpos
        add("04", child_token.pos, direction, distance) #dpos
        add("05", head_token.pos,child_token.pos, direction, distance) #"hpos,dpos"
        add("06", head_token.lemma,head_token.pos, direction, distance) #"hform,hpos"
        add("07", child_token.lemma,child_token.pos, direction, distance) #"dform,dpos"
        add("08", head_token.lemma, head_token.pos, child_token.lemma, child_token.pos, direction, distance) #
        add("09", head_token.pos, child_token.lemma, child_token.pos, direction, distance) #hpos, dform, dpos
        add("10", head_token.lemma, child_token.lemma, child_token.pos, direction, distance) #hform, dform, dpos
        add("11", head_token.lemma, head_token.pos, child_token.lemma, direction, distance) #hform, hpos, dform
        add("12", head_token.lemma, head_token.pos, child_token.pos, direction, distance) #hform, hpos, dpos
        add("13", head_token.lemma, child_token.lemma, direction, distance) #hform, dform
        
        bpos = add_bpos(sentence, head_id, child_id)
        for bp in bpos:
            add("14", head_token.pos, bp, child_token.pos)  #"hpos, bpos, dpos"
        
        if (head_id+1) < len(sentence.tokens) and child_id > 0:
            add("15", head_token.pos, child_token.pos, sentence.tokens[head_id+1].pos, sentence.tokens[child_id-1].pos) #hpos, dpos, hpos+1, dpos-1
        if head_id > 0 and child_id >0:
            add("16", head_token.pos, child_token.pos, sentence.tokens[head_id-1].pos, sentence.tokens[child_id-1].pos) #hpos, dpos, hpos-1, dpos-1
        if (head_id+1) < len(sentence.tokens) and (child_id+1) < len(sentence.tokens):
            add("17", head_token.pos, child_token.pos, sentence.tokens[head_id+1].pos, sentence.tokens[child_id+1].pos) #hpos, dpos, hpos+1, dpos+1
        if head_id > 0  and (child_id+1) < len(sentence.tokens):
            add("18", head_token.pos, child_token.pos, sentence.tokens[head_id-1].pos, sentence.tokens[child_id+1].pos) #hpos, dpos, hpos-1, dpos+1
        
        return None

#%%
class FeatureMapping():
    "object to map all features seen in training data to unique integers"
    def __init__(self, corpus = None):
        self.dict_ = dict()
        if corpus:
            self.make_feature_mapping(corpus)

    def make_feature_mapping(self, corpus):
        "read each sentence in training data, generate all features, map each feature name to an integer"
        
        def make_sent_features(sentence):
            "loop through each position in sentence and generate features from all possible arcs"
            sentence_features = set()
            for i,head_token in enumerate(sentence.tokens):

                for j, child_token in enumerate(sentence.tokens):
                    if j != 0 and j != i: #No arcs come to ROOT and the same position 
                        arc_features = Feature(sentence, i, j)
                        sentence_features.update(arc_features.features)
                        
            return sentence_features
        
        
        for i, sentence in enumerate(corpus.sentences):
            if i % 500 == 0:
                print(f"generating features, {i/len(corpus.sentences)*100:.2f}%")
            #generate all features of a sentence
            sentence_features = make_sent_features(sentence)
            
            #assign an ID for each feature
            for feat in sentence_features:
                if feat in self.dict_: #do not assign ID for features that have been seen
                    continue
                else: #assign an ID for a new feature
                    self.dict_[feat] = len(self.dict_)
        
        return None

#%%
class SparseFeatureVector():
    "based on pre-computed feature_mapping, find active features for an arc"
    def __init__(self, sentence, head_id, child_id, feature_mapping):
        self.active_feats = list()
        self.make_arc_feature_vector(sentence, head_id, child_id, feature_mapping)
  
    def make_arc_feature_vector(self, sentence, head_id, child_id, feature_mapping):
        "for each arc, generate a list of indexes representing the sparse feature vector"
        feature_vector = list()
        arc_features = Feature(sentence, head_id, child_id)
        for feat in arc_features.features: 
            try: 
                feature_vector.append(feature_mapping.dict_[feat]) 
            except KeyError: 
                continue
            
        self.active_feats = sorted(feature_vector)
    
#%%
class Score():
    "object to generate scores for all possible arcs from a sentence"
    def __init__(self, sentence, feature_mapping, w):
        self.n = len(sentence.tokens)
        self.scores = np.empty((self.n,self.n))
        self.scores.fill(-np.inf)
        self.make_score_matrix(sentence, feature_mapping, w)
    
    def make_score_matrix(self, sentence, feature_mapping, w):
        "for each sentence, compute score matrix for all possible arcs"
        def arc_score(feature_vector, w):
            return np.sum(w[feature_vector])
        
        for i, head_token in enumerate(sentence.tokens):
            for j, child_token in enumerate(sentence.tokens):
                if j != 0 and j != i: #No arcs come to ROOT and the same token 
                    arc_feature_vector = SparseFeatureVector(sentence, i, j, feature_mapping)
                    self.scores[i][j] = arc_score(arc_feature_vector.active_feats, w)
    
        return None
