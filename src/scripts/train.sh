python src/train.py --output_dir "./src/models/checkpoints/training_session"           \
                   --content_datapath "./src/data/dummy/content"  \
                   --style_datapath "./src/data/dummy/style"      \
                   --batch_size 5                                \
                   --max_style_train_samples 20                  \
                   --max_content_train_samples 1000              \
                   --max_eval_samples 5                          \
                   --seed 42                                     \
                   --num_train_epochs 20                         \
                   --learning_rate 5e-5                          \
                   --alpha 1.2                                   \
                   --beta 1.8                                    \
                   --gamma 2.3                                   \
                   --delta 4.5                                   \
                   --width 128                                   \
                   --height 128                                  \
                   --crop_width 256                              \
                   --crop_height 256                             \
                   --content_layers_idx 11 17 22 26              \
                   --style_layers_idx 1 3 6 8 11                 \
                   --transformer_size 32                         \
                   --CNN_layer_depth 4                           \
                   --deep_learner


