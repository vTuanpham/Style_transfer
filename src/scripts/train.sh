python src/train.py --output_dir "/models/checkpoints"           \
                   --content_datapath ".src/data/dummy/content"  \
                   --style_datapath ".src/data/dummy/style"      \
                   --batch_size 5                                \
                   --max_style_train_samples 20                  \
                   --max_content_train_samples 1000              \
                   --max_eval_samples 5                          \
                   --seed 42                                     \
                   --num_train_epochs 20                         \
                   --learning_rate 5e-5                          \
                   --alpha 7.8                                   \
                   --beta 12.5                                   \
                   --gamma 14.8                                  \
                   --width 128                                   \
                   --height 128


