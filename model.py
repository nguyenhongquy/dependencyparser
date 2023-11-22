import features, decoder, utils
import random
import numpy as np

#%% MODEL
class Perceptron():
    "Structure Perceptron to guide the decoder towards correct dependency trees"
    def __init__(self, ftmp_len = None):
        if ftmp_len:
            self.weight = np.zeros(ftmp_len)
            self.q = 0 #example encounter
    
    def update(self, sentence, wrong_token, feature_mapping):
        "update weight vector everytime a wrong prediction is made"
        
        #compute feature vectors
        gold_feature_vector = features.SparseFeatureVector(sentence, wrong_token.head, wrong_token.id, feature_mapping)
        pred_feature_vector = features.SparseFeatureVector(sentence, wrong_token.pred_head, wrong_token.id, feature_mapping)
        
        #update weight vector
        self.weight[gold_feature_vector.active_feats] += 1               
        self.weight[pred_feature_vector.active_feats] -= 1 
        return None
    
    def predict(self, test_fp, feature_mapping, output_conll = False, conll_output_fn = "conll_output"):
        "parse a corpus and write prediction in conll format"
        test_corpus = utils.Corpus(test_fp)
        print("predicting dependency trees for held-out set...")
        for i, sentence in enumerate(test_corpus.sentences):
            #for each sentence, compute the score matrix for every possible arc, based on model's weight
            score_matrix = features.Score(sentence, feature_mapping, self.weight)   
            #find the best tree based on current score matrix
            decoder_ = decoder.Decoder(score_matrix.scores)
            pred_tree = decoder_.arcs 
            
            #add pred_head to each token
            for pred_arc in pred_tree:
                child_token = sentence.tokens[pred_arc[1]]
                child_token.pred_head = pred_arc[0]
                
        uas = test_corpus.evaluate()
        print(f"uas: {uas:.2f}")
        
        if output_conll:
            test_corpus.write_conll(conll_output_fn)
            print("wrote output in conll format")
        
    def train(self, epochs, train_corpus, feature_mapping, dev_fp):
        "iteratively predict dependency trees for training examples, then update weights accordingly"
        
        for iter_ in range(1, epochs+1):
            print(iter_)
            random.shuffle(train_corpus.sentences)
            
            for i,sentence in enumerate(train_corpus.sentences):
                if i % 500 == 0: #progress checking
                    print(f"***epoch {iter_}; {i/len(train_corpus.sentences)*100:.2f}%***")
                
                #increment example encounter regardless of update
                self.q += 1
                
                #for each sentence, compute the score matrix for every possible arc, based on current weight
                score_matrix = features.Score(sentence, feature_mapping, self.weight)   
                #find the best tree based on current score matrix
                decoder_ = decoder.Decoder(score_matrix.scores)
                pred_tree = decoder_.arcs 
                
                #add pred_head to each token
                for pred_arc in pred_tree:
                    #each arc is a tuple of (head, child)
                    child_token = sentence.tokens[pred_arc[1]] 
                    child_token.pred_head = pred_arc[0]
                
                #check to update weight
                for idx, token in enumerate(sentence.tokens):
                    if idx != 0: #do not compare ROOT token
                        if token.head != token.pred_head: #wrong prediction
                            #update weight vector
                            self.update(sentence, token, feature_mapping)
            
            #evaluate after each epoch
            uas = train_corpus.evaluate()
            print(f"train uas: {uas:.2f}")
            
            #test each epoch
            self.predict(dev_fp, feature_mapping)

    def featmap_prune(self, threshold, feature_mapping_d, featmap_cutoff_fp):
        "after training, cutoff featmap with zero weight then save the short version of featmap"
        def is_important(item):
            k, v = item
            return abs(self.weight[v]) > threshold
        
        print(f"full featmap {len(feature_mapping_d)}")
        it_ = filter(is_important, feature_mapping_d.items()) #we want to keep only important features in featmap
        short_featmap = dict()
        while True:
            try:
                k,v = next(it_)
                short_featmap[k] = v #copy important features to new featmap
            except StopIteration:
                break
        print(f"short featmap {len(short_featmap)}")
        utils.save(short_featmap, featmap_cutoff_fp)
        print("saved short featmap")
#%%
class AveragedPerceptron(Perceptron):
    "for better generalization, we need to average all weight vectors seen during training and use the averaged weight vector"
    def __init__(self, ftmp_len = None):
        #initialize weight vector, cached weight, example encounter
        if ftmp_len:
            self.weight = np.zeros(ftmp_len)
            self.cached_weight = np.zeros(ftmp_len)
            self.q = 0 #example encounter
    
    def update(self, sentence, wrong_token, feature_mapping):
        "update weight vector and cached weights when a wrong prediction is made"
        #compute feature vectors
        gold_feature_vector = features.SparseFeatureVector(sentence, wrong_token.head, wrong_token.id, feature_mapping)
        pred_feature_vector = features.SparseFeatureVector(sentence, wrong_token.pred_head, wrong_token.id, feature_mapping)
        self.weight[gold_feature_vector.active_feats] += 1               
        self.weight[pred_feature_vector.active_feats] -= 1 
        self.cached_weight[gold_feature_vector.active_feats] += self.q
        self.cached_weight[pred_feature_vector.active_feats] -= self.q
        return None
    
    def average(self):
        "after training, return the averaged weight vector"
        self.weight = self.weight - 1/self.q * self.cached_weight
        print("Averaged weight vector")
        return None
     
