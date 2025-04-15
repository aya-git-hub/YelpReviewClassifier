#!/usr/bin/env python3
import json
import os
import sys


class UsefulReviewFilter:
    """
    This class reads a JSON file (with one JSON record per line),
    filters out reviews where useful + funny + cool is less than 3,
    and writes only the first 1,000,000 valid reviews to an output file.
    """

    def __init__(self, filepath):
        """
        Initialize with the complete path to the JSON file.
        """
        self.filepath = filepath

    def filter_reviews(self, output_filepath, limit=500000):
        """
        Process the input JSON file and write the valid reviews to output_filepath.

        Args:
            output_filepath (str): The path of the output file.
            limit (int): The maximum number of valid reviews to save.
        """
        count = 0
        try:
            with open(self.filepath, 'r', encoding='utf-8') as infile, \
                    open(output_filepath, 'w', encoding='utf-8') as outfile:
                for line_number, line in enumerate(infile, start=1):
                    try:
                        review = json.loads(line)
                    except json.JSONDecodeError as e:
                        # Skip lines that are not valid JSON.
                        print(f"Skipping invalid JSON at line {line_number}: {e}")
                        continue

                    # Get the values for the keys "useful", "funny" and "cool".
                    # If one of these keys is missing, we assume a value of 0.
                    useful = review.get("useful", 0)
                    funny = review.get("funny", 0)
                    cool = review.get("cool", 0)

                    # Check if the sum is at least 1.
                    if (useful + funny + cool) >= 5:
                        outfile.write(json.dumps(review, ensure_ascii=False) + "\n")
                        count += 1
                        if count % 1000 == 0:
                            print(f"Processed {count} valid reviews...")
                        if count >= limit:
                            print(f"Reached limit of {limit} valid reviews.")
                            break

                print(f"Filtering complete. {count} valid reviews saved to {output_filepath}.")
        except FileNotFoundError:
            print(f"Error: The file {self.filepath} was not found.")
        except Exception as e:
            print(f"An error occurred while filtering reviews: {e}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 UsefulReviewFilter.py <path> <json file name>")
        sys.exit(1)

    # Construct full path of the input file.
    directory = sys.argv[1]
    filename = sys.argv[2]
    input_filepath = os.path.join(directory, filename)

    # Define the output file path.
    output_filename = "yelp_useful.json"
    output_filepath = os.path.join(directory, output_filename)

    # Create an instance of UsefulReviewFilter and run the filtering.
    review_filter = UsefulReviewFilter(input_filepath)
    review_filter.filter_reviews(output_filepath)


if __name__ == "__main__":
    main()
