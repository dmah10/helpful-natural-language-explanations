import pandas as pd
from groq import Groq
import time
import logging
import json
import os
from datetime import datetime
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Choose your model here
SELECTED_MODEL = "llama3-70b-8192"  # Change this to "llama3-70b-8192"  or "mixtral-8x7b-32768" or "gemma2-9b-it" for other models
START_INDEX = 0  # Change this if you want to start from a specific index


def initialize_client():
    return Groq(
        api_key="_",  # TODO
    )


def check_rate_limit_error(error_message):
    """Check if error is related to rate limits"""
    rate_limit_keywords = [
        "rate limit",
        "too many requests",
        "quota exceeded",
        "monthly token limit",
        "requests per minute exceeded",
        "tokens per minute exceeded",
    ]
    return any(keyword in error_message.lower() for keyword in rate_limit_keywords)


def make_groq_call(client, message_content):
    try:
        return client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": message_content,
                }
            ],
            model=SELECTED_MODEL,
            temperature=0.0,
        )
    except Exception as e:
        error_message = str(e)
        if check_rate_limit_error(error_message):
            logger.error("Rate limit reached. Saving progress and stopping.")
            raise Exception("RATE_LIMIT_REACHED")
        raise e


def save_progress(index, timestamp):
    """Save the current progress to a JSON file"""
    progress = {"last_processed_index": index, "timestamp": timestamp.isoformat()}
    with open(f"progress_{SELECTED_MODEL.split('-')[0]}.json", "w") as f:
        json.dump(progress, f)


def save_results(chat_completion_responses, current_index, is_final=False):
    """Save results to Excel and backup CSV"""
    df = pd.DataFrame(chat_completion_responses)
    model_name = SELECTED_MODEL.split("-")[0]

    # Save to Excel
    file_suffix = "final" if is_final else f"intermediate_{current_index}"
    excel_file = f"explanations_{model_name}_HFC_{file_suffix}_FS2.xlsx"

    try:
        df.to_excel(excel_file, index=False)
        logger.info(f"Saved results to {excel_file}")
    except Exception as e:
        logger.error(f"Error saving to Excel: {str(e)}")
        # Save to CSV as backup
        csv_file = excel_file.replace(".xlsx", ".csv")
        df.to_csv(csv_file, index=False)
        logger.info(f"Saved backup to {csv_file}")


def process_data():
    client = initialize_client()

    # Load and process the data
    file_path = "healthFC_sentences.xlsx"
    df = pd.read_excel(file_path)

    # Start from the specified index
    df = df.iloc[START_INDEX:]
    df = df.reset_index(drop=True)

    chat_completion_responses = []
    num_examples = len(df)
    BATCH_SIZE = 3  # Small batch size
    PAUSE_SECONDS = 10  # Pause between batches

    logger.info(
        f"Starting processing with model: {SELECTED_MODEL} from index {START_INDEX}"
    )

    try:
        for i, row in df.iterrows():
            current_index = i + START_INDEX

            try:
                en_claim = row["en_claim"]
                en_filtered_sentences = row["en_filtered_sentences"]
                label_num = row["label"]
                en_explanation = row["en_explanation"]

                message_content = (
                    f"Given the following:\n\n"
                    f'Claim: "{en_claim}"\n'
                    f'Sentences: "{en_filtered_sentences}"\n'
                    f'Label: "{label_num}" (where Supported = 0, Not enough information = 1, Refuted = 2)\n\n'
                    "Now, given the provided claim, sentences, and label, provide only the concise explanation in one sentence, directly referencing the claim and the provided sentences. "
                    "Do not include any prefixes like 'Explanation:' or 'Here is the explanation'. "
                    "Start directly with the explanation sentence."
                    "The explanation should not explicitly hint at the label or contain the label itself in any form."
                    "Focus solely on reasoning that connects the premise to the hypothesis without revealing the classification."
                    "\n\n"
                    "Here are some examples:\n\n"
                    'Claim: "Can masks reduce corona infections when worn by a large proportion of the population?"\n'
                    "Sentences: \"['Studies show slow spread After three years of pandemic, there are now relatively meaningful data that masks can slow down the spread of the Corona virus.', "
                    "'Surgical masks, in turn, seem to reduce the risk of infecting themselves with the corona virus.']\"\n"
                    'Label: "0"\n'
                    'Explanation: "International studies suggest that the number of corona infections decreases when many people wear masks. However, the exact protective effect depends on mask type and usage."\n\n'
                    'Claim: "Can homeopathic preparations with Anamirta cocculus help against dizziness?"\n'
                    "Sentences: \"['But without success: We could not find any meaningful scientific evidence for effectiveness in dizziness.', "
                    "'It is now well proven by studies that homeopathic preparations usually do not work better than a sham medication (placebo).']\"\n"
                    'Label: "1"\n'
                    'Explanation: "No reliable evidence supports the effectiveness of homeopathic preparations for dizziness."\n\n'
                )

                chat_completion = make_groq_call(client, message_content)
                explanation = chat_completion.choices[0].message.content

                chat_completion_responses.append(
                    {
                        "en_claim": en_claim,
                        "en_filtered_sentences": en_filtered_sentences,
                        "label": label_num,
                        "explanation": explanation,
                        "human_explanation": en_explanation,
                    }
                )

                logger.info(f"Successfully processed example {current_index + 1}")

                # Batch-based rate limiting
                if (i + 1) % BATCH_SIZE == 0:
                    logger.info(
                        f"Processed {current_index + 1} examples, pausing for {PAUSE_SECONDS} seconds..."
                    )
                    time.sleep(PAUSE_SECONDS)

                # Save intermediate results every 25 examples
                if (i + 1) % 100 == 0:
                    save_results(chat_completion_responses, current_index)
                    save_progress(current_index + 1, datetime.now())
                    logger.info(
                        f"Saved intermediate results at example {current_index + 1}"
                    )

            except Exception as e:
                if str(e) == "RATE_LIMIT_REACHED":
                    logger.error(
                        f"Rate limit reached at index {current_index}. Saving progress..."
                    )
                    save_results(chat_completion_responses, current_index)
                    save_progress(current_index, datetime.now())
                    logger.info(
                        f"To resume, set START_INDEX = {current_index} in the script"
                    )
                    sys.exit(1)

                logger.error(f"Error processing example {current_index + 1}: {str(e)}")
                chat_completion_responses.append(
                    {
                        "en_claim": en_claim,
                        "en_filtered_sentences": en_filtered_sentences,
                        "label": label_num,
                        "explanation": "",
                        "human_explanation": en_explanation,
                    }
                )

        # Save final results
        save_results(chat_completion_responses, len(df), is_final=True)
        logger.info("Processing complete!")

    except KeyboardInterrupt:
        logger.info("\nScript interrupted by user. Saving progress...")
        save_results(chat_completion_responses, current_index)
        save_progress(current_index, datetime.now())
        logger.info(f"To resume, set START_INDEX = {current_index} in the script")
        sys.exit(0)


def main():
    print("Starting the script...")
    process_data()


if __name__ == "__main__":
    main()
