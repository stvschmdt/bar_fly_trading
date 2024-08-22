from PIL import Image, ImageDraw, ImageFont
import os

class SectionedPNGtoPDFConverter:
    def __init__(self, directory, output_pdf):
        self.directory = directory
        self.output_pdf = output_pdf

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
            if 'analysis' in file.lower():
                sections["Stock Analysis"].append(file_path)
            elif 'sector' in file.lower():
                sections["Sector Analysis"].append(file_path)
            elif 'market' in file.lower():
                sections["Market Analysis"].append(file_path)

        # List to hold the image objects
        images = []

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

# Example usage
converter = SectionedPNGtoPDFConverter(directory='plots', output_pdf='output.pdf')
converter.convert()

