# Context-Aware Tag Prediction
This tag engine is devised for recommending relevant tags. This exploits various context information such as image, location, and time to suggest a list of tags. The tags to be output are pre-defined based on how often they occur in the collected Instagram posts.

## Prerequisite (Dependencies)
We use PyTorch ver. 1.4.0 and HuggingFace's Transformers library for our implementation.
```bash
conda install pytorch==1.4.0
conda install pip
pip install .
```
  
## How to Use
After cloning the code to your repository, you can train and evaluate for each model with the following shell script:

For training:
```bash
python examples/language-modeling/run_tag_generation.py \
    --model_type bert \
    --model_name_or_path bert-base-uncased \
    --output_dir models/your_model \
    --do_train \
    --train_data_file dataset/train.json \
    --block_size 384 \
    --tag_list dataset/tagset_rev.txt \
    --use_image \
    --use_location \
    --use_time \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 5 \
    --save_steps 10000 \
    --loss_fct KL \
```
For inference:
```bash
python examples/language-modeling/run_tag_generation.py \
    --output_dir models/your_model \
    --model_type bert \
    --model_name_or_path models/your_model \
    --do_eval \
    --eval_data_file dataset/test.json \
    --block_size 384 \
    --tag_list dataset/tagset_rev.txt \
    --tag2contexts dataset/tag2contexts.json \
    --use_image \
    --use_location \
    --use_time \
```
For evaluation:
```bash
python evaluation.py \
    --result_file models/your_model/results.json \
    --eval_data_file dataset/test.json \
    --tag_list dataset/tagset_rev.txt \
```
You may want to change each script to change some arguments.
