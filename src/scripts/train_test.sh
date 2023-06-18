python src/train.py --output_dir "./src/models/checkpoints/training_session"                      \
                   --content_datapath "./src/data/dummy/content" "./src/data/anotherdummy/content"  \
                   --style_datapath "./src/data/dummy/style" "./src/data/anotherdummy/style"        \
                   --batch_size 2                                \
                   --max_style_train_samples 15                  \
                   --max_content_train_samples 15                \
                   --seed 42                                     \
                   --num_train_epochs 5                          \
                   --learning_rate 1e-4                          \
                   --alpha 1                                     \
                   --beta 1                                      \
                   --gamma 1                                     \
                   --delta 1                                     \
                   --crop_width 32                               \
                   --crop_height 32                              \
                   --content_layers_idx 12 16 21                 \
                   --style_layers_idx 0 5 10 19 28               \
                   --transformer_size 16                         \
                   --CNN_layer_depth 2                           \
                   --deep_learner                                \
                   --step_frequency 0.3                          \
                   --num_worker 2