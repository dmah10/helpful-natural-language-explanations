import pandas as pd
import logging
import json
import os
from datetime import datetime
import time
import sys
from groq import Groq

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Configuration
SELECTED_MODEL = "llama3-70b-8192"  # Change this to "llama3-70b-8192"  or "mixtral-8x7b-32768" or "gemma2-9b-it" for other models
START_INDEX = 0
INPUT_FILE = "balanced_esnli_subset.xlsx"


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
    with open(f"progress_esnli_balanced_{SELECTED_MODEL.split('-')[0]}.json", "w") as f:
        json.dump(progress, f)


def save_results(chat_completion_responses, current_index, is_final=False):
    """Save results to Excel and backup CSV"""
    df = pd.DataFrame(chat_completion_responses)
    model_name = SELECTED_MODEL.split("-")[0]

    # Save to Excel
    file_suffix = "final" if is_final else f"intermediate_{current_index}"
    excel_file = f"explanations_{model_name}_esnli_balanced_{file_suffix}_FS3.xlsx"

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

    # Load balanced dataset
    logger.info(f"Loading balanced dataset from {INPUT_FILE}")
    df = pd.read_excel(INPUT_FILE)

    # Start from specified index
    df = df.iloc[START_INDEX:]
    df = df.reset_index(drop=True)

    chat_completion_responses = []
    BATCH_SIZE = 3
    PAUSE_SECONDS = 10

    logger.info(
        f"Starting processing with model: {SELECTED_MODEL} from index {START_INDEX}"
    )

    try:
        for i, row in df.iterrows():
            current_index = i + START_INDEX

            try:
                premise = row["premise"]
                hypothesis = row["hypothesis"]
                label_num = row["label"]
                hexplanation1 = row["explanation_1"]

                message_content = (
                    f"Given the following:\n\n"
                    f'Premise: "{premise}"\n'
                    f'Hypothesis: "{hypothesis}"\n'
                    f'Label: "{label_num}" (where entailment = 0, neutral = 1, contradiction = 2)\n\n'
                    "Provide exactly one sentence that directly connects the premise to the hypothesis."
                    "Do not include any prefixes like 'Explanation:' or 'Here is the explanation'. "
                    "Start directly with the explanation sentence."
                    "The explanation should not explicitly hint at the label or contain the label itself in any form."
                    "Focus solely on reasoning that connects the premise to the hypothesis without revealing the classification."
                    "\n\n"
                    "Examples:\n\n"
                    "Premise: Person in green gear does trick in midair with motorcycle.\n"
                    "Hypothesis: Person in gear does trick in midair with motorcycle.\n"
                    "Label: 0\n"
                    "A person does trick in midair while wearing green gear.\n\n"
                    "Premise: A man leans against a pay phone while reading a paper.\n"
                    "Hypothesis: The man is standing and holing a newspaper.\n"
                    "Label: 0\n"
                    "If the man is reading a paper he is reading a newspaper.\n\n"
                    "Premise: People walk down a street in a metropolitan area where graffiti can be seen.\n"
                    "Hypothesis: The graffiti is good art\n"
                    "Label: 1\n"
                    "Not all graffiti is good art.\n\n"
                    "Premise: Football players wearing helmets practice defensive moves while their coach stands by ready to blow his whistle.\n"
                    "Hypothesis: The coach drills his team hard\n"
                    "Label: 0\n"
                    "In both sentence, coach are teaching her team.\n\n"
                    "Premise: A young woman in relaxed clothing vacuuming hallway with small vacuum.\n"
                    "Hypothesis: A mother cleans up after her children who spilled cereal in the hallway by vacuuming.\n"
                    "Label: 1\n"
                    "Not all women are a mother with children.\n\n"
                )

                chat_completion = make_groq_call(client, message_content)
                explanation = chat_completion.choices[0].message.content

                chat_completion_responses.append(
                    {
                        "premise": premise,
                        "hypothesis": hypothesis,
                        "label": label_num,
                        "explanation": explanation,
                        "human_explanation": hexplanation1,
                    }
                )

                logger.info(f"Successfully processed example {current_index + 1}")

                # Batch-based rate limiting
                if (i + 1) % BATCH_SIZE == 0:
                    logger.info(
                        f"Processed {current_index + 1} examples, pausing for {PAUSE_SECONDS} seconds..."
                    )
                    time.sleep(PAUSE_SECONDS)

                # Save intermediate results every 100 examples
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
                        "premise": premise,
                        "hypothesis": hypothesis,
                        "label": label_num,
                        "explanation": "",
                        "human_explanation": hexplanation1,
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
