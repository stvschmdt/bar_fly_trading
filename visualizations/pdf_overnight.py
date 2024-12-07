import argparse
import logging
import os

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from util import get_closest_trading_date

logger = logging.getLogger(__name__)


class SectionedPNGtoPDFConverter:
    def __init__(self, directory, output_pdf):
        self.directory = directory
        self.output_pdf = output_pdf
        # parse the date from the directory name overnight_2021-09-01
        self.date = self.directory.split('/')[-1].split('_')[-1]

    def convert(self):
        # Dictionary to hold images by section
        sections = {
            "Stock Analysis": [],
            "Sector Analysis": [],
            "Market Analysis": []
        }

        # Get a list of all PNG files in the directory
        png_files = [f for f in os.listdir(self.directory) if f.endswith('.png')]
        
        # Sort the files to maintain order
        png_files.sort()

        # Categorize images into sections
        for file in png_files:
            file_path = os.path.join(self.directory, file)
            if 'daily' in file.lower() or 'technical' in file.lower():
                sections["Stock Analysis"].append(file_path)
            elif 'sector' in file.lower():
                sections["Sector Analysis"].append(file_path)
            elif 'market' in file.lower():
                sections["Market Analysis"].append(file_path)

        # List to hold the image objects
        images = []
        # first read in the results csv to create a table image as first pdf page
        title_img = self.csv_to_image_table()
        title_img = Image.open('table_image.png').convert('RGB')
        # ensure size is a full normal page
#        title_img = title_img.resize((2500, 2500))
        images.append(title_img)

        # Define a simple font and image size for the section titles
        font = ImageFont.load_default()
        title_image_size = (1000, 100)  # Adjust as needed

        for section, files in sections.items():
            if files:
                # Create a blank image for the section title
                title_image = Image.new('RGB', title_image_size, color='white')
                draw = ImageDraw.Draw(title_image)

                # Calculate text size and position
                text_bbox = draw.textbbox((0, 0), section, font=font)
                text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
                text_position = ((title_image_size[0] - text_width) // 2, (title_image_size[1] - text_height) // 2)
                
                # Draw the text on the title image
                draw.text(text_position, section, fill='black', font=font)
                images.append(title_image)

                for file in files:
                    img = Image.open(file).convert('RGB')
                    images.append(img)

        # Save all images as a PDF
        if images:
            # The first image is the base image, the others are appended
            images[0].save(self.output_pdf, save_all=True, append_images=images[1:])
            print(f'PDF created successfully: {self.output_pdf}')
        else:
            print('No PNG files found in the directory.')


    def csv_to_image_table(self,csv_path='screener_results_', output_image='table_image.png', output_pdf='table_image.pdf'):
        # Load CSV file into DataFrame
        df = pd.read_csv(csv_path+self.date+'.csv')
        # sort by symbol and reset index
        df = df.sort_values(by='symbol').reset_index(drop=True)
        
        # add a column subtracting the num_bullish from num_bearish
        df['bull_bear_delta'] = df['num_bullish'] - df['num_bearish']
        # get a list of the columns and move the bull_bear_delta to second position after symbol
        cols = df.columns.tolist()
        cols = cols[:1] + cols[-1:] + cols[1:-1]
        df = df[cols]
        print(df)

        # Set font and dimensions for the table
        font = ImageFont.load_default()

        padding = 20
        cell_width = 125
        cell_height = 30

        # Calculate image size
        num_columns = len(df.columns)
        num_rows = len(df) + 1  # Adding 1 for headers
        img_width = cell_width * num_columns + padding * 2
        img_height = cell_height * num_rows + padding * 2

        # Create image with white background
        img = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(img)
        # add a title for the image above the table
        title = 'Screener Results for '+self.date
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_width, title_height = title_bbox[2] - title_bbox[0], title_bbox[3] - title_bbox[1]
        title_position = ((img_width - title_width) // 2, padding // 2)
        draw.text(title_position, title, fill='black', font=font)

        # Draw table header
        y_offset = padding
        for col_idx, col_name in enumerate(df.columns):
            x_offset = padding + col_idx * cell_width
            draw.rectangle([x_offset, y_offset, x_offset + cell_width, y_offset + cell_height], outline='black', width=1)
            draw.text((x_offset + 5, y_offset + 15), col_name, fill='black', font=font)

        # Draw table rows
        y_offset += cell_height
        for row_idx, row in df.iterrows():
            for col_idx, cell_value in enumerate(row):
                x_offset = padding + col_idx * cell_width
                draw.rectangle([x_offset, y_offset, x_offset + cell_width, y_offset + cell_height], outline='black', width=1)
                draw.text((x_offset + 5, y_offset + 15), str(cell_value), fill='black', font=font)
            y_offset += cell_height

        # Resize image to fit all rows if needed
        img = img.crop((0, 0, img_width, y_offset + padding))

        # Save the image
        img.save(output_image)


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Convert sectioned PNG files to a single PDF')
    # add directory argument
    parser.add_argument('-d', '--directory', default='overnight_'+get_closest_trading_date(pd.Timestamp.now().strftime('%Y-%m-%d')), help='Directory containing sectioned PNG files')
    # if no directory is provided, try the current date appended to overnight_
    try:
        args = parser.parse_args()
        logger.info('Creating PDF for directory: %s', args.directory)
        # Get date from directory for output PDF name
        output_date = get_closest_trading_date(args.directory.split('_')[-1])
        converter = SectionedPNGtoPDFConverter(directory=args.directory, output_pdf=f'overnight_{output_date}.pdf')
        converter.convert()
    except:
        print('Unable to create PDF')
