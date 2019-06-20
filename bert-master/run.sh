export BERT_BASE_DIR=D:/nlp/bert/chinese_L-12_H-768_A-12
export NER_DIR=C:/github/BERT-NER/tmp
python run_NER.py \
          --task_name=NER \
          --do_train=true \
          --do_eval=true \
          --do_predict=true \
          --data_dir=$NER_DIR/ \
          --vocab_file=$BERT_BASE_DIR/vocab.txt \
          --bert_config_file=$BERT_BASE_DIR/bert_config.json \
          --learning_rate=2e-5 \
          --train_batch_size=32 \
          --num_train_epochs=3 \
          --output_dir=$BERT_BASE_DIR/output \
          --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
          --max_seq_length=256 \     # 根据实际句子长度可调
