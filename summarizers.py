from src.MemSum_Full.model import LocalSentenceEncoder as LocalSentenceEncoder_MemSum_Full
from src.MemSum_Full.model import GlobalContextEncoder as GlobalContextEncoder_MemSum_Full
from src.MemSum_Full.model import ExtractionContextDecoder as ExtractionContextDecoder_MemSum_Full
from src.MemSum_Full.model import Extractor as Extractor_MemSum_Full
from src.MemSum_Full.datautils import Vocab as Vocab_MemSum_Full
from src.MemSum_Full.datautils import SentenceTokenizer as SentenceTokenizer_MemSum_Full


import torch.nn.functional as F
from torch.distributions import Categorical

import pickle
import torch
import numpy as np

from tqdm import tqdm
import json


class MemSum:
    def __init__( self, model_path, vocabulary_path, gpu = None , embed_dim=300, num_heads=8, hidden_dim = 1024, N_enc_l = 2 , N_enc_g = 2, N_dec = 3,  max_seq_len =100, max_doc_len = 500  ):
        # with open( vocabulary_path , 'rb' ) as f:
        #     words = pickle.load(f)
        
        from gensim.models import KeyedVectors
        import gc
        model = KeyedVectors.load_word2vec_format(vocabulary_path, binary=True)
        # self.vocab = Vocab_MemSum_Full( words )
        self.vocab = Vocab_MemSum_Full(model.index_to_key, dict(model.key_to_index))
        
        vocab_size = len(model)
        del model
        gc.collect()

        self.local_sentence_encoder = LocalSentenceEncoder_MemSum_Full( vocab_size, self.vocab.pad_index, embed_dim,num_heads,hidden_dim,N_enc_l, None )
        self.global_context_encoder = GlobalContextEncoder_MemSum_Full( embed_dim, num_heads, hidden_dim, N_enc_g )
        self.extraction_context_decoder = ExtractionContextDecoder_MemSum_Full( embed_dim, num_heads, hidden_dim, N_dec )
        self.extractor = Extractor_MemSum_Full( embed_dim, num_heads )
        ckpt = torch.load( model_path, map_location = 'cpu' )
        self.local_sentence_encoder.load_state_dict( ckpt['local_sentence_encoder'] )
        self.global_context_encoder.load_state_dict( ckpt['global_context_encoder'] )
        self.extraction_context_decoder.load_state_dict( ckpt['extraction_context_decoder'] )
        self.extractor.load_state_dict(ckpt['extractor'])
        
        self.device =  torch.device( 'cuda:%d'%(gpu) if gpu is not None and torch.cuda.is_available() else 'cpu'  )        
        self.local_sentence_encoder.to(self.device)
        self.global_context_encoder.to(self.device)
        self.extraction_context_decoder.to(self.device)
        self.extractor.to(self.device)
        
        self.sentence_tokenizer = SentenceTokenizer_MemSum_Full()
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
    
    def get_ngram(self,  w_list, n = 4 ):
        ngram_set = set()
        for pos in range(len(w_list) - n + 1 ):
            ngram_set.add( '_'.join( w_list[ pos:pos+n] )  )
        return ngram_set

    def extract( self, document_batch, p_stop_thres = 0.7, ngram_blocking = False, ngram = 3, return_sentence_position = False, return_sentence_score_history = False, max_extracted_sentences_per_document = 4 ):
        '''document_batch is a batch of documents:
        [  [ sen1, sen2, ... , senL1 ], 
           [ sen1, sen2, ... , senL2], ...
         ]
        '''
        ## tokenization:
        document_length_list = []
        sentence_length_list = []
        tokenized_document_batch = []
        for document in document_batch:
            tokenized_document = []
            for sen in document:
                tokenized_sen = self.sentence_tokenizer.tokenize( sen )
                tokenized_document.append( tokenized_sen )
                sentence_length_list.append( len(tokenized_sen.split()) )
            tokenized_document_batch.append( tokenized_document )
            document_length_list.append( len(tokenized_document) )

        max_document_length =  self.max_doc_len 
        max_sentence_length =  self.max_seq_len 
        ## convert to sequence
        seqs = []
        doc_mask = []
        
        for document in tokenized_document_batch:
            if len(document) > max_document_length:
                # doc_mask.append(  [0] * max_document_length )
                document = document[:max_document_length]
            else:
                # doc_mask.append(  [0] * len(document) +[1] * ( max_document_length -  len(document) ) )
                document = document + [''] * ( max_document_length -  len(document) )

            doc_mask.append(  [ 1 if sen.strip() == '' else 0 for sen in  document   ] )

            document_sequences = []
            for sen in document:
                seq = self.vocab.sent2seq( sen, max_sentence_length )
                document_sequences.append(seq)
            seqs.append(document_sequences)
        seqs = np.asarray(seqs)
        doc_mask = np.asarray(doc_mask) == 1
        seqs = torch.from_numpy(seqs).to(self.device)
        doc_mask = torch.from_numpy(doc_mask).to(self.device)

        extracted_sentences = []
        sentence_score_history = []
        p_stop_history = []
        
        with torch.no_grad():
            num_sentences = seqs.size(1)
            sen_embed  = self.local_sentence_encoder( seqs.view(-1, seqs.size(2) )  )
            sen_embed = sen_embed.view( -1, num_sentences, sen_embed.size(1) )
            relevance_embed = self.global_context_encoder( sen_embed, doc_mask  )
    
            num_documents = seqs.size(0)
            doc_mask = doc_mask.detach().cpu().numpy()
            seqs = seqs.detach().cpu().numpy()
    
            extracted_sentences = []
            extracted_sentences_positions = []
        
            for doc_i in range(num_documents):
                current_doc_mask = doc_mask[doc_i:doc_i+1]
                current_remaining_mask_np = np.ones_like(current_doc_mask ).astype(bool) | current_doc_mask
                current_extraction_mask_np = np.zeros_like(current_doc_mask).astype(bool) | current_doc_mask
        
                current_sen_embed = sen_embed[doc_i:doc_i+1]
                current_relevance_embed = relevance_embed[ doc_i:doc_i+1 ]
                current_redundancy_embed = None
        
                current_hyps = []
                extracted_sen_ngrams = set()

                sentence_score_history_for_doc_i = []

                p_stop_history_for_doc_i = []
                
                for step in range( max_extracted_sentences_per_document+1 ) :
                    current_extraction_mask = torch.from_numpy( current_extraction_mask_np ).to(self.device)
                    current_remaining_mask = torch.from_numpy( current_remaining_mask_np ).to(self.device)
                    if step > 0:
                        current_redundancy_embed = self.extraction_context_decoder( current_sen_embed, current_remaining_mask, current_extraction_mask  )
                    p, p_stop, _ = self.extractor( current_sen_embed, current_relevance_embed, current_redundancy_embed , current_extraction_mask  )
                    p_stop = p_stop.unsqueeze(1)
            
            
                    p = p.masked_fill( current_extraction_mask, 1e-12 ) 

                    sentence_score_history_for_doc_i.append( p.detach().cpu().numpy() )

                    p_stop_history_for_doc_i.append(  p_stop.squeeze(1).item() )

                    normalized_p = p / p.sum(dim=1, keepdims = True)

                    stop = p_stop.squeeze(1).item()> p_stop_thres #and step > 0
                    
                    #sen_i = normalized_p.argmax(dim=1)[0]
                    _, sorted_sen_indices =normalized_p.sort(dim=1, descending= True)
                    sorted_sen_indices = sorted_sen_indices[0]
                    
                    extracted = False
                    for sen_i in sorted_sen_indices:
                        sen_i = sen_i.item()
                        if sen_i< len(document_batch[doc_i]):
                            sen = document_batch[doc_i][sen_i]
                        else:
                            break
                        sen_ngrams = self.get_ngram( sen.lower().split(), ngram )
                        if not ngram_blocking or len( extracted_sen_ngrams &  sen_ngrams ) < 1:
                            extracted_sen_ngrams.update( sen_ngrams )
                            extracted = True
                            break
                                        
                    if stop or step == max_extracted_sentences_per_document or not extracted:
                        extracted_sentences.append( [ document_batch[doc_i][sen_i] for sen_i in  current_hyps if sen_i < len(document_batch[doc_i])    ] )
                        extracted_sentences_positions.append( [ sen_i for sen_i in  current_hyps if sen_i < len(document_batch[doc_i])  ]  )
                        break
                    else:
                        current_hyps.append(sen_i)
                        current_extraction_mask_np[0, sen_i] = True
                        current_remaining_mask_np[0, sen_i] = False

                sentence_score_history.append(sentence_score_history_for_doc_i)
                p_stop_history.append( p_stop_history_for_doc_i )

        # if return_sentence_position:
        #     return extracted_sentences, extracted_sentences_positions 
        # else:
        #     return extracted_sentences

        results = [extracted_sentences]
        if return_sentence_position:
            results.append( extracted_sentences_positions )
        if return_sentence_score_history:
            results+=[sentence_score_history , p_stop_history ]
        if len(results) == 1:
            results = results[0]
        
        return results



class ExtractiveSummarizer_NeuSum:
    def __init__( self, model_path, vocabulary_path, gpu = None , embed_dim=200,
                 max_seq_len =100, max_doc_len = 500 , **kwargs ):
        with open( vocabulary_path , 'rb' ) as f:
            words = pickle.load(f)
        self.vocab = Vocab_NeuSum( words )
        vocab_size = len(words)
        self.local_sentence_encoder = LocalSentenceEncoder_NeuSum( vocab_size, self.vocab.pad_index, embed_dim, None )
        self.global_context_encoder = GlobalContextEncoder_NeuSum( embed_dim)
        self.extraction_context_decoder = ExtractionContextDecoder_NeuSum( embed_dim)
        self.extractor = Extractor_NeuSum( embed_dim )
        ckpt = torch.load( model_path, map_location = 'cpu' )
        self.local_sentence_encoder.load_state_dict( ckpt['local_sentence_encoder'] )
        self.global_context_encoder.load_state_dict( ckpt['global_context_encoder'] )
        self.extraction_context_decoder.load_state_dict( ckpt['extraction_context_decoder'] )
        self.extractor.load_state_dict(ckpt['extractor'])
        
        self.device =  torch.device( 'cuda:%d'%(gpu) if gpu is not None and torch.cuda.is_available() else 'cpu'  )        
        self.local_sentence_encoder.to(self.device)
        self.global_context_encoder.to(self.device)
        self.extraction_context_decoder.to(self.device)
        self.extractor.to(self.device)
        
        self.sentence_tokenizer = SentenceTokenizer_NeuSum()
        self.max_seq_len = max_seq_len
        self.max_doc_len = max_doc_len
    

    def extract( self, document_batch, return_sentence_position = False, max_extracted_sentences_per_document = 7, **kwargs ):
        '''document_batch is a batch of documents:
        [  [ sen1, sen2, ... , senL1 ], 
           [ sen1, sen2, ... , senL2], ...
         ]
        '''
        ## tokenization:
        document_length_list = []
        sentence_length_list = []
        tokenized_document_batch = []
        for document in document_batch:
            tokenized_document = []
            for sen in document:
                tokenized_sen = self.sentence_tokenizer.tokenize( sen )
                tokenized_document.append( tokenized_sen )
                sentence_length_list.append( len(tokenized_sen.split()) )
            tokenized_document_batch.append( tokenized_document )
            document_length_list.append( len(tokenized_document) )

        max_document_length =  self.max_doc_len 
        max_sentence_length =  self.max_seq_len 
        ## convert to sequence
        seqs = []
        doc_mask = []
        
        for document in tokenized_document_batch:
            if len(document) > max_document_length:
                # doc_mask.append(  [0] * max_document_length )
                document = document[:max_document_length]
            else:
                # doc_mask.append(  [0] * len(document) +[1] * ( max_document_length -  len(document) ) )
                document = document + [''] * ( max_document_length -  len(document) )

            doc_mask.append(  [ 1 if sen.strip() == '' else 0 for sen in  document   ] )

            document_sequences = []
            for sen in document:
                seq = self.vocab.sent2seq( sen, max_sentence_length )
                document_sequences.append(seq)
            seqs.append(document_sequences)
        seqs = np.asarray(seqs)
        doc_mask = np.asarray(doc_mask) == 1
        seqs = torch.from_numpy(seqs).to(self.device)
        doc_mask = torch.from_numpy(doc_mask).to(self.device)
        
        with torch.no_grad():
            num_sentences = seqs.size(1)
            sen_embed  = self.local_sentence_encoder( seqs.view(-1, seqs.size(2) )  )
            sen_embed = sen_embed.view( -1, num_sentences, sen_embed.size(1) )
            global_context_embed, backward_state = self.global_context_encoder( sen_embed, doc_mask,  return_backward_state = True  )
    
            num_documents = seqs.size(0)
            doc_mask = doc_mask.detach().cpu().numpy()
            seqs = seqs.detach().cpu().numpy()
    
            extracted_sentences = []
            extracted_sentences_positions = []
        
            for doc_i in range(num_documents):
                current_doc_mask = doc_mask[doc_i:doc_i+1]
                current_remaining_mask_np = np.ones_like(current_doc_mask ).astype(bool) | current_doc_mask
                current_extraction_mask_np = np.zeros_like(current_doc_mask).astype(bool) | current_doc_mask
        
                current_global_context_embed = global_context_embed[doc_i:doc_i+1]
                current_hidden_state = backward_state[ doc_i:doc_i+1 ]
                current_extracted_sen_embed =  torch.zeros_like(  current_global_context_embed[:,:1,:] )
        
                current_hyps = []

                for step in range( max_extracted_sentences_per_document ) :
                    current_extraction_mask = torch.from_numpy( current_extraction_mask_np ).to(self.device)
                    current_remaining_mask = torch.from_numpy( current_remaining_mask_np ).to(self.device)
                    current_hidden_state  = self.extraction_context_decoder( current_extracted_sen_embed, current_hidden_state )
                    p = self.extractor( current_global_context_embed, current_hidden_state, current_extraction_mask )
                                                            
                    sen_i = p.argmax(dim=1)[0]
                    sen_i = sen_i.item()
                    
                    current_hyps.append(sen_i)
                    current_extraction_mask_np[0, sen_i] = True
                    current_remaining_mask_np[0, sen_i] = False
                    
                                        
                extracted_sentences.append( [ document_batch[doc_i][sen_i] for sen_i in  current_hyps if sen_i < len(document_batch[doc_i])    ] )
                extracted_sentences_positions.append( [ sen_i for sen_i in  current_hyps if sen_i < len(document_batch[doc_i])  ]  )


        results = [extracted_sentences]
        if return_sentence_position:
            results.append( extracted_sentences_positions )
        if len(results) == 1:
            results = results[0]
        return results

from tqdm import tqdm
from rouge_score import rouge_scorer
import json
import numpy as np
import os
import nltk

def run_eval():
    rouge_cal = rouge_scorer.RougeScorer(['rouge1','rouge2', 'rougeLsum'], use_stemmer=True)

    memsum = MemSum('model/rulingBR/300dim/run0/model_batch_63720.pt', 'model/glove/glove_s300.bin', gpu=0, max_doc_len=500)
    test_corpus = [ json.loads(line) for line in open('data/custom_data/val_CUSTOM_raw.jsonl') ]

    print(evaluate(memsum, test_corpus, 0.6, 25, rouge_cal))

def evaluate(model, corpus, p_stop, max_extracted_sentences, rouge_cal):
    scores = []
    for data in tqdm(corpus):
        gold_summary = data["summary"]
        extracted_summary = model.extract( [data["text"]], p_stop_thres = p_stop, max_extracted_sentences_per_document = max_extracted_sentences )[0]
        
        score = rouge_cal.score( "\n".join( gold_summary ), "\n".join(extracted_summary)  )
        scores.append( [score["rouge1"].fmeasure, score["rouge2"].fmeasure, score["rougeLsum"].fmeasure ] )
    
    return np.asarray(scores).mean(axis = 0)

def run_predict():
    max_doc_len=500
    p_stop = 0.6
    max_extracted_sentences = 5

    model_name = 'model/rulingBR/300dim/run0/model_batch_63720.pt'
    print(f'max_doc_len: {max_doc_len}')
    print(f'max_extracted_sentences: {max_extracted_sentences}')
    print(f'p_stop: {p_stop}')
    print(f'Loading model {model_name}')
    memsum = MemSum(model_name, 'model/glove/glove_s300.bin', gpu=0, max_doc_len=max_doc_len)
    

    # filename = 'instrucao2.txt'
    filename = 'instrucao_tratada_exame_tecnico.txt'
    # filename = 'exame_tecnico_teste.txt'
    # filename = 'instrucao_tratada_exame_tecnico_media.txt'
    # filename = 'texto_aleatorio.txt'
    # filename = 'instrucao_ruim.txt'

    path = os.path.join('../abs-sum-pt/tcu/instrucoes_questionario')
    with open(os.path.join(path, filename), 'r',  encoding='utf-8') as f:
        example = f.read()
    
    print(f'Arquivo: {filename}')
    print(predict(memsum, example, p_stop=p_stop, max_extracted_sentences=max_extracted_sentences))

def predict(model, example, p_stop, max_extracted_sentences):
    stok = nltk.data.load('tokenizers/punkt/portuguese.pickle')
    example = stok.tokenize(example.lower())
    extracted_summary = model.extract([example], p_stop_thres = p_stop, max_extracted_sentences_per_document = max_extracted_sentences )[0]
    return '\n'.join(extracted_summary)



if __name__ == '__main__':
    # run_eval()
    run_predict()