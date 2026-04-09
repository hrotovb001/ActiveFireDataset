import csv
import re
import os
from datetime import datetime, timedelta
import argparse
import boto3
from botocore import UNSIGNED
from botocore.config import Config


def load_fire_dates(csv_path):
    dates = set()
    with open(csv_path, mode="r", newline="") as file:
        csv_reader = csv.reader(file, delimiter=",")
        next(csv_reader)
        for row in csv_reader:
            dt = datetime.strptime(f"{row[5]} {row[6]}", "%Y-%m-%d %H%M")
            lng = float(row[1])
            dates.add((dt, lng))
    return dates


def download_goes_files(dates, output_dir):
    s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
    bucket_18 = s3.Bucket("noaa-goes18")
    bucket_19 = s3.Bucket("noaa-goes19")

    for date, lng in sorted(dates):
        date_adjusted = date - timedelta(minutes=10)
        year = date_adjusted.year
        doy = date_adjusted.timetuple().tm_yday
        hour = date_adjusted.hour

        bucket =  bucket_18 if lng < -110 else bucket_19
        prefix = f"ABI-L2-FDCC/{year}/{doy:03d}/{hour:02d}"
        for obj in bucket.objects.filter(Prefix=prefix):
            match = re.search(r"s(\d+)_e(\d+)", obj.key)
            start_raw, end_raw = match.groups()
            start = datetime.strptime(start_raw, "%Y%j%H%M%S%f")
            end = datetime.strptime(end_raw, "%Y%j%H%M%S%f")

            if start <= date_adjusted <= end:
                print(f"Downloading: {obj.key}")
                local_path = os.path.join(output_dir, obj.key)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                bucket.download_file(obj.key, local_path)
                exit()


def main():
    parser = argparse.ArgumentParser(
        description="Download GOES ABI-L2-FDCC files related to given fire event CSV."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to the input CSV file containing fire data."
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Directory to save the downloaded files."
    )

    args = parser.parse_args()

    fire_dates = load_fire_dates(args.input)
    download_goes_files(fire_dates, args.output)


if __name__ == "__main__":
    main()

