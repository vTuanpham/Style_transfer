python src/train.py --output_dir "./src/models/checkpoints"           \
                   --content_datapath "./src/data/dummy/content"  \
                   --style_datapath "./src/data/dummy/style"      \
                   --batch_size 1                                \
                   --max_style_train_samples 5                   \
                   --max_content_train_samples 10                \
                   --max_eval_samples 5                          \
                   --seed 42                                     \
                   --num_train_epochs 10                         \
                   --learning_rate 5e-5                          \
                   --alpha 0.01                                  \
                   --beta 0.01                                   \
                   --gamma 0.01                                  \
                   --width 32                                    \
                   --height 32
