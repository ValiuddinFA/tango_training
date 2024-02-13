import os
import requests
import pickle
import zipfile
import logging
import gensim

from constant.CONSTANT import BASE_URL, USER_NAME, PASSWORD, ENCRYPTION_KEY, ALT_ENCRYPTION_KEY, LOCAL_FILE_PATH, DOWNLOAD_LINKS, META_DOWNLOADS

class Downloader:
    def __init__(self, url, save_directory, chunk_size=128, auth=False):
        self.url = url
        self.save_directory = save_directory
        self.chunk_size = chunk_size
        self.auth = auth

    def download_file(self):
        """Download file from url to save_path in chunks."""
        # Prepare authentication
        auth_params = None
        if self.auth:
            auth_params = (USER_NAME, PASSWORD)

        # Ensure the save directory exists
        os.makedirs(self.save_directory, exist_ok=True)

        # Get the file name from the URL
        file_name = self.url.split('/')[-1]

        # Construct the save path
        save_path = os.path.join(self.save_directory, file_name)

        # Download file in chunks
        with requests.get(self.url, stream=True, auth=auth_params) as response:
            response.raise_for_status()
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=self.chunk_size):
                    # Write the chunk to the file
                    f.write(chunk)
                    # Yield the chunk
                    yield chunk


class ZipExtractor:
    def __init__(self, file_path, pwd):
        self.file_path = file_path
        self.pwd = pwd

    def extract_zip(self):
        """Extract zip file"""
        with zipfile.ZipFile(self.file_path) as zip_ref:
            zip_ref.extractall('.', pwd=self.pwd.encode())
        return zip_ref.namelist()


class FirstImpressionsDownloader:
    def __init__(self, server, ENCRYPTION_KEY, meta_downloads, download_links):
        self.server = server
        self.ENCRYPTION_KEY = ENCRYPTION_KEY
        self.meta_downloads = meta_downloads
        self.download_links = download_links

    def download_first_impressions(self):
        """Download raw video and annotations datasets. Extract the zipped folders and organize them."""
        logging.info('Attempting to either validate or download first impressions dataset.')

        # Load dictionaries that map relationships between downloaded files
        with open('resources/file_tree.pkl', 'rb') as input_file:
            file_tree = pickle.load(input_file)
        with open('resources/meta_tree.pkl', 'rb') as input_file:
            meta_tree = pickle.load(input_file)

        # Check if meta files are downloaded
        for file in self.meta_downloads:
            fi_downloaded_path = os.path.join('resources/resources/compressed', file)
            if not os.path.exists(fi_downloaded_path):
                logging.warning(f'{file} does not exist...downloading')
                yield from self._download_file(self.server + file, fi_downloaded_path, auth=True)

        # Check if the meta files are in the correct location
        for file in meta_tree.keys():
            if not os.path.exists(os.path.join('./data/meta_data', file)):
                # Extract and save to the correct location using the encryption key
                with zipfile.ZipFile(os.path.join('resources/resources/compressed', meta_tree[file]), 'r') as zip_ref:
                    yield from zip_ref.extractall('./data/meta_data/', pwd=self.ENCRYPTION_KEY.encode())

        for file in self.download_links:
            # Define path to compressed files
            fi_downloaded_path = os.path.join('resources/resources/compressed', file)

            # Download files, if necessary
            if not os.path.exists(fi_downloaded_path):
                logging.warning('{} does not exist. Downloading {}.'.format(file, file))
                logging.info('Downloading embedding data from: {} to: {}'.format(self.server + file, fi_downloaded_path))

                # Download the file
                yield from self._download_file(self.server + file, fi_downloaded_path, auth=True)

        for file_chunk in self.download_links:
            # Define path to compressed files
            fi_downloaded_path = os.path.join('resources/resources/compressed', file_chunk)
            # Extract and save to the correct location using the encryption key
            with zipfile.ZipFile(fi_downloaded_path, 'r') as zip_ref:
                yield from zip_ref.extractall('./data/video_data/', pwd=self.ENCRYPTION_KEY.encode())

    def _download_file(self, url, save_path, chunk_size=128, auth=False):
        """
        Download file from url to save_path in chunks.
        """
        logging.info(f'Downloading file from {url} to {save_path}')

        # Prepare authentication
        auth_params = None
        if auth:
            auth_params = (USER_NAME, PASSWORD)

        # Define the generator function
        def download_chunks():
            with requests.get(url, stream=True, auth=auth_params) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=chunk_size):
                    yield chunk

        # Download file in chunks using the generator
        with open(save_path, 'wb') as f:
            for chunk in download_chunks():
                logging.debug(f'Downloaded chunk: {chunk}')
                # Configure logging
                logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
                f.write(chunk)
                print(f'Received chunk of size {len(chunk)}')
                print(f'Downloaded {save_path}')

        return save_path

    def __init__(self, model_path='./resources/GoogleNews-vectors-negative300.bin'):
        self.model_path = model_path

    def create_embedding_matrix(self):
        """
        Load embedding assets from file.
         - Load embedding binaries w/ gensim
         - Extract embedding matrix from gensim model
         - Extract word to index lookup from gensim model
        :return: embedding_matrix, word_to_index
        :rtype: (numpy.array, {str:int})
        """

        logging.info('Reading embedding matrix and word to index dictionary from file')

        # Get word weights from file via gensim
        model = gensim.models.KeyedVectors.load_word2vec_format(self.model_path, binary=True)

        embedding_matrix = model.vectors
        word_to_index = {word: index for index, word in enumerate(model.index_to_key)}

        logging.info('Created embedding matrix, of shape: {}'.format(embedding_matrix.shape))
        logging.info('Created word to index lookup, with min index: {}, max index: {}'.format(min(word_to_index.values()),
                                                                                              max(word_to_index.values())))

        return embedding_matrix, word_to_index


if __name__ == "__main__":
    # Example usage:
   

    # Downloader
    save_directory = './data/'
    for url in DOWNLOAD_LINKS:
        full_url = BASE_URL + url
        downloader = Downloader(full_url, save_directory)
        for chunk in downloader.download_file():
            print(f'Received chunk of size {len(chunk)}')
        print(f'Downloaded {full_url}')

    # ZipExtractor
    file_path = './data/file.zip'
    pwd = 'password'
    zip_extractor = ZipExtractor(file_path, pwd)
    zip_extractor.extract_zip()

    # FirstImpressionsDownloader
    server = BASE_URL
    first_impressions_downloader = FirstImpressionsDownloader(server, ENCRYPTION_KEY, META_DOWNLOADS, DOWNLOAD_LINKS)  
    for result in first_impressions_downloader.download_first_impressions():
        print(result)
