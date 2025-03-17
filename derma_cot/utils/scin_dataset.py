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


def create_context(row: dict[str, Any]):
    # Parse skin conditions and weights
    conditions = eval(row["skin_condition"])  # Convert string list to actual list
    weights = eval(row["weighted_skin_label"])  # Convert string dict to actual dict

    # 1. Skin Condition Section
    if conditions:
        condition_str = (
            "Dermatological inspection of the physical characteristics of the subject in the image indicate that it could be the following conditions with the specified probabilities: "
            + ", ".join([f"{cond} ({weights[cond] * 100:.0f}%)" for cond in conditions])
            + "."
        )
    else:
        condition_str = "The patient has submitted this image with details and symptoms reported below."

    # 2. Demographic Section
    age_mapper = {
        "AGE_UNKNOWN": "Unknown",
        "AGE_18_TO_29": "18 to 29",
        "AGE_40_TO_49": "40 to 49",
        "AGE_50_TO_59": "50 to 59",
        "AGE_70_TO_79": "70 to 79",
        "AGE_60_TO_69": "60 to 69",
        "AGE_30_TO_39": "30 to 39",
        "nan": None,
    }

    sex_mapper = {
        "OTHER_OR_UNSPECIFIED": "Unknown",
        "MALE": "Male",
        "FEMALE": "Female",
        "nan": None,
    }

    fitz_mapper = {
        "FST2": "Type II",
        "FST1": "Type I",
        "FST4": "Type IV",
        "nan": None,
        "FST3": "Type III",
        "FST5": "Type V",
        "FST6": "Type VI",
    }

    monk_mapper = {
        "1.0": "A (Light tone with reference hex colour code #f7ede4)",
        "2.0": "B (Light tone with reference hex colour code #f3e7da)",
        "3.0": "C (Light tone with reference hex colour code #f6ead0)",
        "4.0": "D (Medium tone with reference hex colour code #ead9bb)",
        "5.0": "E (Medium tone with reference hex colour code #d7bd96)",
        "6.0": "F (Medium tone with reference hex colour code #9f7d54)",
        "7.0": "G (Dark tone with reference hex colour code #815d44)",
        "8.0": "H (Dark tone with reference hex colour code #604234)",
        "9.0": "I (Dark tone with reference hex colour code #3a312a)",
        "10.0": "J (Dark tone with reference hex colour code #2a2420)",
        "nan": None,
    }

    demo_parts = []
    fitz = fitz_mapper[row["fitzpatrick_type"]]
    monk = monk_mapper[row["monk_skin_tone"]]
    sex = sex_mapper[row["sex"]]
    age = age_mapper[row["age_group"]]
    if fitz not in [None, ""]:
        if not monk:
            demo_parts.append(f"Fitzpatrick skin type: {fitz}")
    if monk not in [None, ""]:
        demo_parts.append(f"Monk skin tone: {monk}")
    if sex not in ["Unknown", None]:
        demo_parts.append(f"Sex: {sex}")
    if age not in ["Unknown", None]:
        demo_parts.append(f"Age group: {age}")

    demo_str = (
        "Patient demographic details are as follows: " + ". ".join(demo_parts) + "."
        if demo_parts
        else ""
    )

    # 3. Self-reported Features Section
    symptom_groups = {
        "Skin Texture of affected parts": [
            k for k in row if k.startswith("self_textures_") and row[k]
        ],
        "Body Parts with symptoms": [
            k for k in row if k.startswith("self_body_parts_") and row[k]
        ],
        "Symptoms": [
            k for k in row if k.startswith("self_condition_symptoms_") and row[k]
        ],
        "Other Symptoms": [
            k for k in row if k.startswith("self_other_symptoms_") and row[k]
        ],
    }

    symptom_strs = []
    for group_name, features in symptom_groups.items():
        if features:
            clean_features = [
                f.replace("self_textures_", "")
                .replace("self_body_parts_", "")
                .replace("self_condition_symptoms_", "")
                .replace("self_other_symptoms_", "")
                .replace("_", " ")
                .title()
                for f in features
            ]
            symptom_strs.append(f"{group_name}: {', '.join(clean_features)}")

    symptom_str = "\n".join(symptom_strs) if symptom_strs else ""

    # Combine all sections
    final_context = f"{condition_str}\n{demo_str}\n{symptom_str}".strip()
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
                             - Demographic correlations
                             - Reported symptoms
                             Begin your answer with an in-depth description of the physical features of this image as they pertain to a dermatological analysis.
                             If there are no provided differential diagnosis probabilities, then only make conservative suggestions about the diagnosis.""",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                },
            ],
        },
    ]


# def analyze_skin(row: dict[str, Any], client: OpenAI):
#     try:
#         response = client.chat.completions.create(
#             model="Qwen/Qwen2-VL-72B-Instruct",
#             temperature=0,
#             messages=create_messages(row),
#             max_tokens=5000,
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         print(f"Error processing case {row['case_id']}: {str(e)}")
#         return None


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
        case_id = row.get("case_id", "unknown")
        print(f"\nError processing case {case_id}: {str(e)}")
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


def get_scin_labels(
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
    print(f"Loading the SCIN dataset from {filepath}...")
    scin = load_from_disk(filepath)

    # If in trial mode, only use a subset of the dataset
    if trial_mode:
        print(f"Running in trial mode with {trial_size} samples")
        scin = scin.select(range(min(trial_size, len(scin))))  # type: ignore

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
    all_labels = [""] * len(scin)

    print(f"Processing {len(scin)} samples with {max_workers} concurrent workers...")

    # Process all samples with progress bar
    progress_bar = tqdm(total=len(scin), desc="Processing samples")

    # Process in batches to allow checkpointing
    for batch_start in range(0, len(scin), batch_size):
        batch_end = min(batch_start + batch_size, len(scin))
        batch_indices = list(range(batch_start, batch_end))
        batch_rows = [scin[idx] for idx in batch_indices]

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
            checkpoint_dataset = scin.add_column("label", all_labels)  # type: ignore
            checkpoint_path = output_path / f"checkpoint_{batch_start}"
            print(f"\nSaving checkpoint to {checkpoint_path}...")
            checkpoint_dataset.save_to_disk(checkpoint_path)

    progress_bar.close()

    # Add the labels column to the dataset
    final_dataset = scin.add_column("label", all_labels)  # type: ignore

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
    scin = get_scin_labels(
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
    return scin


if __name__ == "__main__":
    main()
