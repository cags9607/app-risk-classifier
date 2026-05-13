import argparse
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder_to_zip", required=True)
    ap.add_argument("--output_zip_path", default=None)
    args = ap.parse_args()

    folder_to_zip = args.folder_to_zip
    output_zip_path = args.output_zip_path
    if output_zip_path is None:
        zip_file_name = folder_to_zip.replace("/content/", "").replace("/", "_")
        output_zip_path = os.path.join("/content/", zip_file_name)

    print(f"Zipping folder: {folder_to_zip} to {output_zip_path}.zip")
    shutil.make_archive(output_zip_path, "zip", folder_to_zip)
    print(f"Zip file created: {output_zip_path}.zip")


if __name__ == "__main__":
    main()
