python tagger.py  \
    --corpus=ACE2005 \
    --train_data=../data/10.04.2021/processed/seq2seq/train.txt \
    --dev_data=../data/10.04.2021/processed/seq2seq/dev.txt \
    --test_data=../data/10.04.2021/processed/seq2seq/test.txt \
    --decoding=seq2seq \
    --epochs=15:1e-3,15:1e-4 \
    --form_wes_model=../pretrained/cc.ru.300.vec.txt \
    --lemma_wes_model=../pretrained/cc.ru.300.vec.txt \
    --name=seq2seq+cc300ru
    # --bert_embeddings_train=bert_embeddings/conll_en_train_dev_bert_large_embeddings.txt \
    # --bert_embeddings_test=bert_embeddings/conll_en_test_bert_large_embeddings.txt \
    # --flair_train=flair_embeddings/conll_en_train_dev.txt \
    # --flair_test=flair_embeddings/conll_en_test.txt \
    # --elmo_train=elmo_embeddings/conll_en_train_dev.txt \
    # --elmo_test=elmo_embeddings/conll_en_test.txt \