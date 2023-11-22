class Token():
    "Object to represent a token"
     
    def __init__(self, id_, form, lemma, pos, head, rel):
        "each attribute is provided from data (in conll format)"
        self.id = int(id_) if id_ != "_" else id_
        self.form = form
        self.lemma = lemma
        self.pos = pos
        self.head = int(head) if head != "_" else head
        self.rel = rel
        self.pred_head = 0 #will be used for evaluation during training
        self.pred_rel = "_" 

        
    def __str__(self):
        fields = [str(self.id), self.form, self.lemma, self.pos, "_", "_", str(self.pred_head), self.rel, "_", "_"] #used to write output in conll format
        return ("\t".join(fields))


class Sentence():
    "Class of Sentence objects, which host a list of tokens"

    def __init__(self, tokens):
        self.tokens = tokens 
        
    def __str__(self):
        sent_text = []
        for token in self.tokens:  
            sent_text.append((token.form))
        return str(sent_text)
    
    def count_correct_token(self):
        "count number of correct heads and correct relation labels"
        correct_head = 0
        correct_rel = 0
        nr_tok = len(self.tokens)
        for tok in self.tokens:
            if tok.head != "_":
                if int(tok.head) == tok.pred_head:
                    correct_head += 1
                    if tok.rel == tok.pred_rel:
                        correct_rel += 1
        
        return correct_head, correct_rel, nr_tok

class Corpus():
    "Object to host list of Sentence objects and evaluate training performance"

    def __init__(self, file_path = None):
        self.sentences = []
        if file_path:
            self.read_conll(file_path)
    
    def read_conll(self, file_path):
        "read the corpus file, extract lexical information and gold labels"

        with open(file_path) as f:
            tokens = [Token(0, "ROOT", "LROOT", "PROOT", "_", "_")] #add ROOT token
            line = f.readline()
            while line != '':
                if line == '\n': #each sentence is separated by a new line
                    self.sentences.append(Sentence(tokens)) 
                    tokens = [Token(0, "ROOT", "_", "PROOT", "_", "_")] #add ROOT token to each sentence
                else:
                    l = line.split()
                    id_ = l[0]
                    form = l[1]
                    lemma = l[2]
                    pos = l[3]
                    head = l[6]
                    rel = l[7]
                    tokens.append(Token(id_, form, lemma, pos, head, rel))
                line = f.readline()
                
        return None
    

    def read_pred(self, gold_path, pred_path):
        "read the gold and pred file, add pred_head and pred_rel"

        with open(gold_path) as f1:
            with open(pred_path) as f2:
                tokens = []
                line1 = f1.readline()
                line2 = f2.readline()
                while line1 != '':
                    if line1 == '\n':
                        self.sentences.append(Sentence(tokens))
                        tokens = [] #add ROOT token to each sentence
                    else:
                        l1 = line1.split()
                        l2 = line2.split()
                        id_ = l1[0]
                        form = l1[1]
                        lemma = l1[2]
                        pos = l1[3]
                        head = l1[6]
                        rel = l1[7]
                        tok = Token(id_, form, lemma, pos, head, rel)
                        tok.pred_head = l2[6]
                        tok.pred_rel = l2[7]
                        tokens.append(tok)
                    line1 = f1.readline()
                    line2 = f2.readline()
                
        return None

    def evaluate_from_file(self, gold_path, pred_path):
        "take in the gold file and the automatic predicted file path, evaluate the quality of prediction"
        self.read_pred(gold_path, pred_path)
        sum_correct_head = 0
        sum_correct_rel = 0
        sum_nr_tok = 0
        for sent in self.sentences:
            correct_head, correct_rel, nr_tok = sent.count_correct_token()
            sum_correct_head += correct_head
            sum_correct_rel += correct_rel
            sum_nr_tok += nr_tok
        
        uas = sum_correct_head/sum_nr_tok * 100
        las = sum_correct_rel/sum_nr_tok * 100
        
        return uas, las 
    
    def evaluate(self):
        "after add_pred, evaluate uas"
        sum_correct_head = 0
        sum_correct_rel = 0
        sum_nr_tok = 0
        for sent in self.sentences:
            correct_head, correct_rel, nr_tok = sent.count_correct_token()
            sum_correct_head += correct_head
            sum_correct_rel += correct_rel
            sum_nr_tok += nr_tok
        
        uas = sum_correct_head/sum_nr_tok * 100
        return uas
        
    def write_conll(self, file_path):
        "write the corpus object in conll format"
        with open(file_path, 'w') as f:
            for s in self.sentences:
                tokens = s.tokens
                tokens.pop(0) #remove the ROOT token
                for t in tokens:
                    line = str(t)
                    f.write(line)
                    f.write("\n")
                f.write("\n")  


#%%SOME TRIVIAL TESTS

def test_token():
    tok1 = Token(100, "he", "he", "HE", 4, "ha")
    assert str(tok1) == "100	he	he	HE	_	_	0	ha	_	_"
    print("test Token finished")

def test_io():
    gold_path = "../data/toy.txt"
    pred_path = "../output/output.conll"
    import difflib
     
    with open(gold_path) as file_1:
        file_1_text = file_1.readlines()
     
    with open(pred_path) as file_2:
        file_2_text = file_2.readlines()
     
    # Find and print the diff:
    for line in difflib.unified_diff(
            file_1_text, file_2_text, fromfile='toy.txt',
            tofile='output.conll', lineterm=''):
        print("*****",line)
    print("test IO finished")

#%%dev.pred test
def test_eval():
    gold_path = "../data/english/dev/wsj_dev.conll06.gold"
    pred_path = "../data/english/dev/wsj_dev.conll06.pred" 
    pred_corpus = Corpus()
    uas, las = pred_corpus.evaluate_from_file(gold_path, pred_path)
    assert (uas - 89.74) < 0.01 and (las - 88.11) < 0.01
    print("test eval finished")

# if __name__ == "__main__":
#     test_token()
#     test_io()
#     test_eval()
#%%HELP FUNCTIONS
import _pickle as cPickle 
import gzip 

def save(obj, filename, protocol = -1):
    """Save an object to a compressed disk file.
       Works well with huge objects.
    """
    file = gzip.GzipFile(filename, 'wb')
    cPickle.dump(obj, file, protocol)
    file.close()
    print("saved weight")
    
def load(filename):
    """Loads a compressed object from disk
    """
    file = gzip.GzipFile(filename, 'rb')
    obj = cPickle.load(file)
    file.close()

    return obj