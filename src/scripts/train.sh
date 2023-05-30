python train.py --output_dir "/models/checkpoints" \
                 --content_datapath "./src/data/style_data/Data/Artworks" \
                 --style_datapath "./src/data/style_data/Data/TestCases" \
                 --batch_size 1        \
                 --max_train_samples 20\
                 --max_eval_samples 5  \
                 --seed 42             \
                 --num_train_epochs 10 \
                 --learning_rate 5e-5  \
                 --alpha 0.01          \
                 --beta 0.01           \
                 --gamma 0.01



