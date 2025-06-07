import argparse
import os
from tqdm import tqdm
import json
from zero_shot_inference.utils import (
    load_test_file,
    encode_image_base64,
    load_prompt,
    call_openai_api,
    save_prediction,
)

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "predictions.jsonl")

    image_list = load_test_file(args.test_file, args.dataset)
    if args.max_images:
        image_list = image_list[:args.max_images]
    prompt = load_prompt(args.prompt_file)

    for image_path, image_id in tqdm(image_list):
        try:
            image_b64 = encode_image_base64(image_path)
            prediction = call_openai_api(image_b64, prompt, model=args.model)
            try:
                prediction_obj = json.loads(prediction)
            except json.JSONDecodeError:
                print(f"[WARNING] Failed to parse prediction for {image_id}. Saving raw.")
                prediction_obj = prediction
            save_prediction(output_file, image_id, prediction_obj)
        except Exception as e:
            print(f"[ERROR] Failed for {image_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--max_images", type=int, default=None, help="Limit number of images for testing")
    args = parser.parse_args()
    main(args)
