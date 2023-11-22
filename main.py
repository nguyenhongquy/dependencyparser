#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:27:01 2023

@author: quynguyen
"""

from optparse import OptionParser
import time
import utils, features, model
import numpy as np
#english = "/mount/studenten/dependency-parsing/data/english" #IMS
#german = "/mount/studenten/dependency-parsing/data/german"

#%%

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--train", dest="train_fp", help="Annotated CONLL train file", metavar="FILE", default="../data/english/train/wsj_train.only-projective.conll06")
    parser.add_option("--dev", dest="dev_fp", help="Annotated CONLL dev file", metavar="FILE", default="../data/english/dev/wsj_dev.conll06.gold")
    parser.add_option("--test", dest="test_fp", help="Annotated CONLL test file", metavar="FILE", default="../data/english/test/wsj_test.conll06.blind")
    parser.add_option("--weights", dest="weight_fp", help="compressed .npz archive", metavar="FILE", default="../averaged_perceptron_english/weight.npz")
    parser.add_option("--featmap", dest= "featmap_cutoff_fp", help="compressed feature mapping file", metavar="FILE", default="../averaged_perceptron_english/feat_map_cutoff")
    parser.add_option("--epochs", type="int", dest="epochs", default=5)
    parser.add_option("--disableAveraged", action="store_false", dest="averaged_flag", default = True)
    parser.add_option("--predict", action="store_true", dest="predict_flag", default=False)
    parser.add_option("--outdir", type="string", dest="conll_output_fn", default="conll_output_pred")
    
    (options, args) = parser.parse_args()
    
    if options.predict_flag:
        print("Testing mode enabled")
        feat_map = features.FeatureMapping()
        feat_map.dict_ = utils.load(options.featmap_cutoff_fp)
        print(f"Feature Mapping loaded, {len(feat_map.dict_)} features")
        dictdata = np.load(options.weight_fp)
        avr_perceptron = model.AveragedPerceptron()
        avr_perceptron.weight = dictdata['arr_0']
        print("Loaded pretrained model")
        avr_perceptron.predict(options.test_fp, feat_map, output_conll = True)
    else:
        print("Training mode enabled")
        start = time.time()
        train_corpus = utils.Corpus(options.train_fp) 
        feature_mapping = features.FeatureMapping(train_corpus)
        avr_perceptron = model.AveragedPerceptron(len(feature_mapping.dict_))
        avr_perceptron.train(options.epochs, train_corpus, feature_mapping, options.dev_fp)
        if options.averaged_flag:
            avr_perceptron.average()
        avr_perceptron.predict(options.dev_fp, feature_mapping)
        np.savez_compressed(options.weight_fp, avr_perceptron.weight)
        print(f"finished training in {(time.time()-start):.2f}s")
        avr_perceptron.featmap_prune(0.0, feature_mapping.dict_, options.featmap_cutoff_fp)

    
