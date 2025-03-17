import base64
import concurrent.futures
import os
import time
from argparse import ArgumentParser
from io import BytesIO
from pathlib import Path
from typing import Any

from datasets import load_from_disk
from openai import OpenAI
from PIL import Image
from ratelimit import limits, sleep_and_retry
from tqdm.auto import tqdm


def create_context(row):
    # Extract core medical information
    disease = row["disease"].strip().title()
    caption = row["caption"].strip()

    # 1. Disease Identification Section
    disease_str = f"Diagnosis: {disease}"

    # 2. Clinical Description Section
    desc_str = f"Clinical Presentation: {caption}"

    # # 3. Additional Context Section
    #  context_parts = []
    #
    # # Handle skin tone mapping if available
    # if str(row['skin_tone']) not in ['nan', 'None', '']:
    #     skin_tone = row['skin_tone']
    #     if skin_tone.startswith('FST'):
    #         context_parts.append(f"Fitzpatrick Skin Type: {skin_tone.replace('FST', 'Type ')}")
    #     else:
    #         context_parts.append(f"Skin Tone: {skin_tone}")
    #
    # # Add data source if relevant
    # if str(row['source']) not in ['nan', 'None', '']:
    #     source_map = {
    #         'fitzpatrick17k': 'Fitzpatrick 17k dataset',
    #         'dermnet': 'DermNet NZ',
    #         'ddi': 'DDI Dataset'
    #     }
    #     clean_source = source_map.get(row['source'].lower(), row['source'])
    #     context_parts.append(f"Source: {clean_source}")
    #
    context_str = ""
    # if context_parts:
    #     context_str = "\nAdditional Context: " + "; ".join(context_parts)

    # Combine all sections
    final_context = f"{disease_str}\n{desc_str}\n{context_str}"

    # Handle special remark cases
    if str(row["remark"]) not in ["nan", "None", ""]:
        final_context += f"\nClinician Remark: {row['remark'].capitalize()}"

    return final_context


def image_to_base64(pil_image: Image.Image):
    """Convert PIL image to base64 string"""
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return base64_image


def create_messages(row: dict[str, Any]):
    # Generate context
    context = create_context(row)

    # Convert image to base64
    base64_image = image_to_base64(row["image"])

    return [
        {
            "role": "system",
            "content": """You are an expert model trained at identifying medically significant features used for dermatological diagnosis of images of human skin.
                You need to generate a detailed response based on the provided image and description.
                Use the background information provided by analysing the image and reading the text provided to assist in formulating a relevant and detailed answer.

                Follow these answer guidelines:
                1. Utilize the details observed in the image to comprehensively understand the physical condition of the human subject's skin in the image.
                2. Utilize the text content containing medically relevant information to provide a comprehensive and accurate answer.
                3. Ensure proper formatting and readability, including the correct rendering of any LaTeX or mathematical symbols.
                4. Please make sure to detail the provided patient demographic information, and how it bears a relation to the presented ailment.
                5. Ensure that you only generate a response that is a detailed description of the image and how it ties into the provided text context.
                5. If there are no provided diagnosis details in the provided text context, do not make any assumptions, and instead simply provide a detailed description of the image.""",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""Analyze the image of a human patient's skin and provide a detailed description,
                             making use of the details found in:\n\n{context}\n\nConsider the following aspects:
                             - Primary visual characteristics
                             - Differential diagnosis probabilities
                             - Reported symptoms
                             Begin your answer with an in-depth description of the physical features of this image as they pertain to a dermatological analysis.
                             If there is no provided differential diagnosis, then only make conservative suggestions about the diagnosis.""",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        },
    ]


def analyze_skin(row: dict[str, Any], client: OpenAI):
    """
    Get AI response for a single image.

    Args:
        row: Dataset row containing image and metadata
        client: OpenAI client

    Returns:
        AI response text or None if an error occurred
    """
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2-VL-72B-Instruct",
            temperature=0,
            messages=create_messages(row),
            max_tokens=5000,
        )
        return response.choices[0].message.content
    except Exception as e:
        skincap_id = row.get("skincap_file_path", "unknown")
        print(f"\nError processing case {skincap_id}: {str(e)}")
        # If rate limit error, suggest waiting
        if "rate limit" in str(e).lower():
            print("Rate limit exceeded. Waiting for 60 seconds before retrying...")
            time.sleep(60)
            try:
                # Retry once
                response = client.chat.completions.create(
                    model="Qwen/Qwen2-VL-72B-Instruct",
                    temperature=0,
                    messages=create_messages(row),
                    max_tokens=5000,
                )
                return response.choices[0].message.content
            except Exception as retry_e:
                print(f"Retry failed: {str(retry_e)}")
        return None


@sleep_and_retry
@limits(calls=1, period=3)  # Default: 20 requests per minute = 1 request per 3 seconds
def rate_limited_analyze_skin(row: dict[str, Any], client: OpenAI):
    return analyze_skin(row, client)


def process_sample(args: tuple[int, dict, OpenAI, int]):
    idx, row, client, max_retries = args
    retries = 0
    while retries <= max_retries:
        try:
            response = rate_limited_analyze_skin(row, client)
            return idx, response
        except Exception as e:
            retries += 1
            if retries <= max_retries:
                print(f"\nRetrying {idx} ({retries}/{max_retries}): {str(e)}")
                time.sleep(5 * retries)
            else:
                print(
                    f"\nFailed after {max_retries} retries for sample {idx}: {str(e)}"
                )
                return idx, None


def get_skincap_labels(
    filepath: Path,
    output_path: Path | Any = None,
    hf_repo_id: str | Any = None,
    batch_size: int = 10,
    rate_limit_rpm: int = 20,
    trial_mode: bool = False,
    trial_size: int = 10,
    max_workers: int = 5,  # Number of concurrent workers
    max_retries: int = 3,  # Maximum number of retries per sample
):
    # Load dataset from disk
    print(f"Loading the SkinCAP dataset from {filepath}...")
    scap = load_from_disk(filepath)

    # If in trial mode, only use a subset of the dataset
    if trial_mode:
        print(f"Running in trial mode with {trial_size} samples")
        scap = scap.select(range(min(trial_size, len(scap))))  # type: ignore

    # Initialize an OpenAI client
    client = OpenAI(
        base_url="https://api.studio.nebius.com/v1/",
        api_key=os.environ.get("NEBIUS_API_KEY", "IMAMECHANIC"),
    )

    # Configure rate limiter based on rate_limit_rpm
    # Update the decorator's period parameter
    rate_limited_analyze_skin.__wrapped__.__dict__["_rate_limit_period"] = (  # type: ignore
        60.0 / rate_limit_rpm
    )

    # Create a list to store all labels
    all_labels = [""] * len(scap)

    print(f"Processing {len(scap)} samples with {max_workers} concurrent workers...")

    # Process all samples with progress bar
    progress_bar = tqdm(total=len(scap), desc="Processing samples")

    # Process in batches to allow checkpointing
    for batch_start in range(0, len(scap), batch_size):
        batch_end = min(batch_start + batch_size, len(scap))
        batch_indices = list(range(batch_start, batch_end))
        batch_rows = [scap[idx] for idx in batch_indices]

        # Prepare arguments for concurrent processing
        process_args = [
            (idx, row, client, max_retries)
            for idx, row in zip(batch_indices, batch_rows)
        ]

        # Process batch concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(process_sample, args): args[0]  # type: ignore
                for args in process_args  # type: ignore
            }

            for future in concurrent.futures.as_completed(future_to_idx):
                idx, response = future.result()  # type: ignore
                all_labels[idx] = response if response else ""
                progress_bar.update(1)

        # Save checkpoint after each batch
        if output_path:
            checkpoint_dataset = scap.add_column("label", all_labels)  # type: ignore
            checkpoint_path = output_path / f"checkpoint_{batch_start}"
            print(f"\nSaving checkpoint to {checkpoint_path}...")
            checkpoint_dataset.save_to_disk(checkpoint_path)

    progress_bar.close()

    # Add the labels column to the dataset
    final_dataset = scap.add_column("label", all_labels)  # type: ignore

    # Save final dataset locally if output_path is provided
    if output_path:
        final_path = output_path / "final"
        print(f"Saving final dataset to {final_path}...")
        final_dataset.save_to_disk(final_path)

    # Push to Huggingface if repo_id is provided
    if hf_repo_id:
        print(f"Pushing dataset to Huggingface repository: {hf_repo_id}...")
        final_dataset.push_to_hub(hf_repo_id)
        print("Dataset successfully pushed to Huggingface!")

    print("Processing completed successfully!")
    return final_dataset


def main():
    """Command line interface for processing the SCIN dataset"""
    parser = ArgumentParser(description="Process SCIN dataset and get AI responses")
    parser.add_argument(
        "--filepath", type=Path, required=True, help="Path to the SCIN dataset"
    )
    parser.add_argument(
        "--output_path", type=Path, help="Path to save the processed dataset"
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        help="Huggingface repository ID to push the dataset to",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Number of samples to process before saving a checkpoint",
    )
    parser.add_argument(
        "--rate_limit_rpm",
        type=int,
        default=20,
        help="Rate limit for API requests per minute",
    )
    parser.add_argument(
        "--trial",
        action="store_true",
        help="Run in trial mode with a limited number of samples",
    )
    parser.add_argument(
        "--trial_size",
        type=int,
        default=10,
        help="Number of samples to process in trial mode",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=5,
        help="Maximum number of concurrent workers",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum number of retries per sample",
    )

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if args.output_path:
        args.output_path.mkdir(parents=True, exist_ok=True)

    # Process the dataset
    scap = get_skincap_labels(
        filepath=args.filepath,
        output_path=args.output_path,
        hf_repo_id=args.hf_repo_id,
        batch_size=args.batch_size,
        rate_limit_rpm=args.rate_limit_rpm,
        trial_mode=args.trial,
        trial_size=args.trial_size,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
    )

    print("Processing completed successfully!")
    return scap


if __name__ == "__main__":
    main()
