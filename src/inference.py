import sys
sys.path.insert(0,r'./') #Add root directory here
import os
import torch
import argparse
from random import random
from PIL import Image
from src.models.generator import Encoder, Decoder
from src.models.transformer import MTranspose
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



# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'
encoder = Encoder().eval().to(device)
decoder = Decoder().eval().to(device)

transformer = MTranspose(matrix_size=32).to(device)

checkpoint = torch.load(r"./src/models/checkpoints/TotalL_421.02252197265625_Content_5.210962295532227_Style_296.22662353515625/transformer.pth")
transformer.load_state_dict(checkpoint)

content_image = Image.open(r'C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\style_transfer\src\data\mirflickr\im99.jpg').convert('RGB')
style_image = Image.open(r'C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\style_transfer\src\data\style_data\Data\Artworks\888440.jpg').convert('RGB')
trans = transforms.Compose([
            transforms.ToTensor()
        ])
content_tensor = trans(content_image).to(device)
style_tensor = trans(style_image).to(device)
transformer.eval()
encode_Cfeatures = encoder(content_tensor.unsqueeze(0))
encode_Sfeatures = encoder(style_tensor.unsqueeze(0))
transformed_features = transformer(encode_Cfeatures, encode_Sfeatures)
decode_img = decoder(transformed_features)
plot_image(decode_img.detach().cpu())
plot_image(style_tensor.unsqueeze(0).detach().cpu())
# Convert the tensor back to an image
final_image = transforms.ToPILImage()(decode_img.squeeze(0).detach().cpu())

# Save the final stylized image
final_image.save("stylized_image1.jpg")