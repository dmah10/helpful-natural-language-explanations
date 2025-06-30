import pandas as pd
import os
import glob


def convert_xlsx_to_csv():
    # Get all xlsx files in current directory
    xlsx_files = glob.glob("*.xlsx")

    if not xlsx_files:
        print("No xlsx files found in current directory.")
        return

    for xlsx_file in xlsx_files:
        try:
            # Get the filename without extension
            filename_without_ext = os.path.splitext(xlsx_file)[0]

            # Read the Excel file
            excel_file = pd.read_excel(xlsx_file, sheet_name=None)

            # Convert each sheet to a separate CSV
            for sheet_name, df in excel_file.items():
                # Create CSV filename with sheet name if multiple sheets exist
                if len(excel_file) > 1:
                    csv_filename = f"{filename_without_ext}_{sheet_name}.csv"
                else:
                    csv_filename = f"{filename_without_ext}.csv"

                # Save to CSV
                df.to_csv(csv_filename, index=False)
                print(
                    f"Successfully converted '{xlsx_file}' sheet '{sheet_name}' to '{csv_filename}'"
                )

        except Exception as e:
            print(f"Error converting {xlsx_file}: {str(e)}")


if __name__ == "__main__":
    convert_xlsx_to_csv()
