export PYTHONPATH="$PWD"
DATA_DIR="/home/denilv/projects/ner-benchmarks/data/25.01.2021/processed/mrc"
BERT_DIR="/home/denilv/projects/ner-benchmarks/pretrained/rubert-base-cased"

BERT_DROPOUT=0.1
MRC_DROPOUT=0.4
LR=1e-5
SPAN_WEIGHT=0.1
WARMUP=0
MAXLEN=128
MAXNORM=1.0

OUTPUT_DIR="./logs/mrc-test-run/ru_ner_test_wwmlarge_sgd_warm${WARMUP}lr${LR}_drop${MRC_DROPOUT}_norm${MAXNORM}_bsz32_gold_span_weight${SPAN_WEIGHT}_warmup${WARMUP}_maxlen${MAXLEN}_continue"
CHECKPOINT="/home/denilv/projects/mrc-for-flat-nested-ner/logs/mrc-test-run/ru_ner_test_wwmlarge_sgd_warm0lr1e-5_drop0.4_norm1.0_bsz32_gold_span_weight0.1_warmup0_maxlen128/epoch=2_v2.ckpt"
mkdir -p $OUTPUT_DIR

python trainer.py \
--data_dir $DATA_DIR \
--pretrained_checkpoint $CHECKPOINT \
--bert_config_dir $BERT_DIR \
--max_length $MAXLEN \
--batch_size 6 \
--gpus="0," \
--precision=16 \
--progress_bar_refresh_rate 1 \
--lr $LR \
--distributed_backend=ddp \
--val_check_interval 0.25 \
--accumulate_grad_batches 1 \
--default_root_dir $OUTPUT_DIR \
--mrc_dropout $MRC_DROPOUT \
--bert_dropout $BERT_DROPOUT \
--max_epochs 5 \
--span_loss_candidates "pred_and_gold" \
--weight_span $SPAN_WEIGHT \
--warmup_steps $WARMUP \
--max_length $MAXLEN \
--gradient_clip_val $MAXNORM \
--optimizer "adamw"
