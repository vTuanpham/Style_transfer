python src/train.py --output_dir "./src/models/checkpoints/training_session"                       \
                   --content_datapath "./src/data/dummy/content" "./src/data/anotherdummy/content" \
                   --style_datapath "./src/data/dummy/style" "./src/data/anotherdummy/style"       \
                   --batch_size 5                                \
                   --max_style_train_samples 1000                \
                   --max_content_train_samples 1000              \
                   --max_eval_samples 5                          \
                   --seed 42                                     \
                   --num_train_epochs 20                         \
                   --learning_rate 1e-4                          \
                   --alpha 1.2                                   \
                   --beta 1.8                                    \
                   --gamma 2.3                                   \
                   --delta 4.5                                   \
                   --crop_width 256                              \
                   --crop_height 256                             \
                   --content_layers_idx 12 16 21                 \
                   --style_layers_idx 0 5 10 19 28               \
                   --transformer_size 32                         \
                   --CNN_layer_depth 2                           \
                   --deep_learner                                \
                   --with_tracking                               \
                   --log_weights_cpkt                            \
                   --step_frequency 0.5                          \
                   --num_worker 4


