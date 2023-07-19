import cv2
import os
import argparse


def remove_faces_from_images(input_folder, output_folder):
    """
    Remove faces from images in the input folder and save the modified images in the output folder.

    Args:
        input_folder (str): Path to the input folder containing images.
        output_folder (str): Path to the output folder to save modified images.
    """

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over the images in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            # Read the image
            image = cv2.imread(input_path)

            # Convert the image to grayscale for face detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            # Remove faces from the image
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), -1)

            # Save the modified image without faces
            cv2.imwrite(output_path, image)

            print(f"Face removed from {input_path} and saved to {output_path}")

    print("Face removal completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove faces from images in a folder.')
    parser.add_argument('input_folder', default='input',
                        help='Path to the input folder containing images')
    parser.add_argument('output_folder', default='output',
                        help='Path to the output folder to save modified images')

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder

    remove_faces_from_images(input_folder, output_folder)
