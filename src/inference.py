import sys
sys.path.insert(0,r'./') #Add root directory here
import os
import torch
import argparse
from random import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from src.models.generator import Encoder, Decoder
from src.models.transformer import MTranspose
from src.models.trainer import Trainer
from src.utils.image_plot import plot_image
import torchvision.transforms as transforms


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--path_to_save_cpkt', type=str, help="Path to the save directory json file")
    parser.add_argument('--test_files', nargs='+', help="Path to the test sample json file")
    parser.add_argument('--interactive', action="store_true", help="Whether to enable interactive mode")
    parser.add_argument('--test_batch_size', type=int, default=6, help="Batch size of test dataloader")
    parser.add_argument('--max_test_samples', type=int, default=None, help="Sample size of the test dataset")
    parser.add_argument('--seed', type=int, default=42, help="Seed for dataloader shuffle")

    args = parser.parse_args(args)

    return args
#
#
# def main(args):
#     args = parse_args(args)
#
#     if args.interactive:
#         while True:
#             sentence = input("Prompt: ")
#             if sentence == 'exit':
#                 break
#             input_tokens = tokenizer(sentence, return_tensors="pt")
#             output = model.generate(input_tokens["input_ids"],
#                                     attention_mask=input_tokens["attention_mask"],
#                                     **gen_kwargs)
#
#             decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
#             print("Output: ", decoded_output)
#     else:
#
#         dataloader_args = {
#             "tokenizer": tokenizer,
#             "text_column": 'prompt',
#             "target_column": 'output',
#             "test_file": args.test_files,
#             "val_batch_size": args.test_batch_size,
#             "max_predict_samples": args.max_test_samples,
#             "source_prefix": "Instruction: ",
#             "seed": args.seed
#         }
#         dataloaders = StateDataloader(**dataloader_args)
#         # Metric
#         metrics_name = ["rouge","bleu"]
#         accelerator = Accelerator()
#         model, test_dataloader = accelerator.prepare(model, dataloaders.__call__()['test'])
#         evaluator = Evaluation(metrics_name=metrics_name, dataloader=test_dataloader, gen_kwargs=gen_kwargs)
#
#         result = evaluator.eval(accelerator, tokenizer, model, 'test')
#         prompts, predictions, labels = [], [], []
#         for key, value in result['examples'].items():
#             prompts.append(value['prompt'])
#             predictions.append(value['prediction'])
#             labels.append(value['label'])
#
#     df = pd.DataFrame({'Prompt': prompts,
#                        'Prediction': predictions,
#                        'Label': labels
#                        })
#     csv_out = os.path.join(args.csv_output, 'result.xlsx')
#     df.to_excel(csv_out)
#
#
# if __name__ == "__main__":
#     main(sys.argv[1:])



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder = Encoder().eval().to(device)
decoder = Decoder().eval().to(device)
transformer = MTranspose(matrix_size=32).to(device)

checkpoint = torch.load(r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\style_transfer\src\models\checkpoints\training_session\trans_size32\TotalL_6.053393363952637_Content_1.5243605375289917_Style_2.729487419128418\transformer80000.pth")
transformer.load_state_dict(checkpoint['model_state_dict'])
transformer.eval()

Trainer.plot_comparison(encoder, decoder, transformer,
                        r'C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\style_transfer\src\data\mirflickr\im404.jpg',
                        r'C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\style_transfer\src\data\Art_by_number\train_9\9079.jpg',
                        transforms.Compose([
                            transforms.Resize(256),
                            transforms.ToTensor()
                        ]),
                        device,
                        sleep=20)