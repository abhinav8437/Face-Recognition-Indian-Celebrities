#PARSE IMAGE FROM GOOGLE AND SAVE THEM IN CURRENT DIRECTORY
from Paths import path_to_save_downloaded_images
import sys

def parse_images_of_celebrities_from_google(celebrities,num_of_images):
    sys.path.append('/Users/abhinavrohilla/')
    from google_images_download.google_images_download import googleimagesdownload
    response = googleimagesdownload()
    arguments = {"keywords":celebrities,"limit":num_of_images,"print_urls":False}   #creating list of arguments
    #THIS WILL DOWNLOAD IMAGES TO THE GIVEN LOCATION
    os.chdir(path_to_save_downloaded_images)
    response.download(arguments)   #passing the arguments to the function