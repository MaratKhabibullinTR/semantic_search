from pathlib import Path
import logging
import json
import os

logger = logging.getLogger(__name__)


def convert_json_dataset_to_txt(input_dir: Path, out_dir: Path):
    """
    Reads all .json files from a directory, combines 'title' and 'text' attributes,
    and creates new .txt files with the combined content.
    """
    logger.info("Start converting...")

    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            json_filepath = os.path.join(input_dir, filename)
            
            try:
                with open(json_filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                title = data.get('title', '')
                text = data.get('text', '')

                content = f"{title}\n\n{text}"

                txt_filename = os.path.splitext(filename)[0] + '.txt'
                txt_filepath = os.path.join(out_dir, txt_filename)

                with open(txt_filepath, 'w', encoding='utf-8') as f:
                    f.write(content)

                logger.info(f"Successfully processed {filename} -> {txt_filename}")

            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Error processing {filename}: {e}")
