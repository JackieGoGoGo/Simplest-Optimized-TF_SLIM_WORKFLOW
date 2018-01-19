python3 train_image_classifier.py \  
--train_dir= ./datas/model \  
--dataset_name=xxxx \  
--dataset_split_name=train \  
--dataset_dir= ./datas \  
--model_name=inception_v2 \  
--checkpoint_path=./checkpoint/inception_v2.ckpt \  
--checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits \  
--trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits  \  
--max_number_of_steps=4000 \  
--batch_size=32 \  
--learning_rate=0.0002 \  
--learning_rate_decay_type=fixed \  
--save_interval_secs=60 \  
--save_summaries_secs=60 \  
--log_every_n_steps=100 \  
--optimizer=rmsprop \  
--weight_decay=0.00004 \  
--clone_on_cpu=True



python train_image_classifier.py --train_dir= ./datas/model --dataset_name=xxxx --dataset_split_name=train --dataset_dir=./datas --model_name=inception_v2 --checkpoint_path=./checkpoint/inception_v2.ckpt --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits --max_number_of_steps=4000 --batch_size=32 --learning_rate=0.0002 --learning_rate_decay_type=fixed --save_interval_secs=60 --save_summaries_secs=60 --log_every_n_steps=100 --optimizer=rmsprop --weight_decay=0.00004 --clone_on_cpu=True