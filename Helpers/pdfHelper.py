import os
from pypdf import PdfReader
from pdf2image import convert_from_path
from numpy import array
from doctr.models import ocr_predictor
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

class PdfHelper:

    @staticmethod
    def get_pdf_names_from_folder(dataset):
        """
        It helps to get the pdf names from a folder.
        :param dataset: str, folder path
        :return: list, pdf names
        """
        pdf_paths = []
        for pdf_path in os.listdir(dataset):
            if pdf_path.endswith('.pdf'):
                pdf_paths.append(os.path.join(dataset, pdf_path))
        return pdf_paths
    @staticmethod
    def extract_text_from_image(pdf_path):
        """
        It helps to extract text from PDF.
        :param pdf_path: str, pdf file path
        :return: str, extracted text from PDF
        """
        try:
            pages = convert_from_path(pdf_path, dpi=300)
        except Exception:
            print(Exception)
        try:
            pages = convert_from_path(pdf_path, dpi=300, poppler_path=f'poppler-24.08.0/Library/bin')
        except:
            print("Poppler not found at /poppler-24.08.0/Library/bin")
            print("Returning none")
            return ""
        text = ""
        images = [array(page) for page in pages]
        model = ocr_predictor(pretrained=True)
        result = model(images)
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = " ".join([word.value for word in line.words])
                    text += line_text + "\n"

        return text

    @staticmethod
    def extract_text_from_pdf(pdf_path, save_output=True):
        """
        It extracts text from selected pdf file.

        :param pdf_path: str, path of the PDF file
        :param save_output: boolean, flag to save extracted text.
        :return: str, the extracted text from PDF
        """
        output_file = pdf_path.replace(".pdf", ".txt")
        if save_output and os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                text = [line.strip() for line in f.readlines()]
            text = "\n".join(text)
        else:
            temp_flag = False
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
                # If pdf is don't have a text
                if len(text.replace("\n", "")) < 100:
                    temp_flag = True
                    break
            if temp_flag:
                text = PdfHelper.extract_text_from_image(pdf_path)
            if save_output:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text + "\n")
        return text

    @staticmethod
    def extract_text_from_pdfs(pdf_paths, save_output=True):
        """
        It extracts text from selected pdf files.

        :param pdf_paths: list, list of paths to PDF files
        :param save_output: boolean, flag to save extracted text.
        :return: dict, dictionary of extracted text from PDFs
        """
        text_dict = {}
        for pdf_path in pdf_paths:

            text_dict[pdf_path] = PdfHelper.extract_text_from_pdf(pdf_path, save_output=save_output)

        return text_dict

    @staticmethod
    def chunk_text(text, chunk_size=1000, overlap=200):
        """
        It helps to chunk text into chunks of given size.
        :param text: str, text to chunk
        :param chunk_size: int, chunk size
        :param overlap: int, overlap size
        :return: list, list of chunks
        """
        sentences = nltk.sent_tokenize(text)
        chunks = []
        chunk = []
        char_count = 0
        for sentence in sentences:
            if char_count + len(sentence) > chunk_size:
                chunks.append(" ".join(chunk))
                chunk = chunk[-overlap//len(sentence):]
                char_count = sum(len(s) for s in chunk)
            chunk.append(sentence)
            char_count += len(sentence)
        if chunk:
            chunks.append(" ".join(chunk))
        return chunks

    @staticmethod
    def chunk_texts(texts, chunk_size=1000, overlap=200):
        """
        It helps to chunk texts into chunks of given size.

        :param texts: dict, dictionary of extracted text from PDFs
        :param chunk_size: int, chunk size
        :param overlap: int, overlap size
        :return:
            list, list of chunks
            list, list of metadata
        """
        all_chunks = []
        all_metadata = []
        for path, text in texts.items():
            chunks = PdfHelper.chunk_text(text, chunk_size, overlap)
            for idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({'source_path': path, 'chunk_id': idx})
        return all_chunks, all_metadata