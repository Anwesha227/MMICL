# Run Inference
python -m zero_shot_inference.inference \
  --dataset semi-aves \
  --test_file data/semi-aves/test.txt \
  --prompt_file zero_shot_inference/prompts/species_prompt_1.txt \
  --output_dir results \
  --model gpt-4o-mini

# Run Inference on a Subset of Images for Testing
python -m zero_shot_inference.inference \
  --dataset semi-aves \
  --test_file data/semi-aves/test.txt \
  --prompt_file zero_shot_inference/prompts/species_prompt_1.txt \
  --output_dir results_debug \
  --model gpt-4o-mini \
  --max_images 3