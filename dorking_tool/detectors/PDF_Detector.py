# detectors/PDF_Detector.py
# 
# This is the most complex detector, so I am writing this long guide.
#   First Step: 
#       We try to detect TCs (Turkish Identity Numbers) in the texts.
#       IF we find a TC in the text, the process is completed
#       If no TC is found in the text, we proceed to second step.
#   Second Step: 
#       First, we open the target PDF in LibreOffice.
#       We check if there are any images in the PDF. If no images are found, the process ends without results.
#       If an image is found, we create a blank LibreOffice document and paste the image into it.
#       After completing the pasting operation, we now have a new LibreOffice document filled with images.
#       Then, we perform OCR on the document. If a TC is detected in any of the images, we take a screenshot of that page and we not look other images time saving
# 
# detectors/PDF_Detector.py
# detectors/PDF_Detector.py

import os
import shutil
import argparse
import logging
from io import BytesIO
import fitz  # PyMuPDF
import requests
from PIL import Image
import cv2
import numpy as np
import easyocr
from urllib3.util import Retry
import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import pdfplumber
from detectors.util import (
    is_valid_tc_kimlik_no,
    download_file,
    get_main_domain,
    ensure_url_scheme,
    is_valid_url
)
import urllib3
from requests.adapters import HTTPAdapter
from urllib.parse import urlparse
import subprocess  # LibreOffice komutları için
import tempfile  # Geçici dosyalar için
from reportlab.pdfgen import canvas  # PDF oluşturmak için
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import base64  # Base64 kodlama için

# Disable InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Load environment variables
import tldextract
from dotenv import load_dotenv
load_dotenv()

# Constants
THREAD_COUNT = int(os.getenv('pdf_detector_THREAD_COUNT', 4))  # Örnek: 4 çalışan
USE_GPU = os.getenv('pdf_detector_USE_GPU', 'False').lower() in ['true', '1', 't']


FAST_ONLY = os.getenv("PDF_FAST_ONLY", "true").lower() in ["1", "true"]

def fast_text_tc_scan(pdf_path):
    """
    PDF’yi sadece pdfplumber ile açıp metni tarar.
    TC bulunursa (found, count) döner.
    """
    try:
        import pdfplumber, re
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
        nums = re.findall(r"(?<!\\d)\\d{11}(?!\\d)", text)
        valid = [n for n in nums if is_valid_tc_kimlik_no(n)]
        return bool(valid), len(valid)
    except Exception as e:
        logger.error(f"Fast scan failed: {e}")
        return False, 0

# Determine project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

from dorking_tool.logger_setup import setup_logger
logger = setup_logger()

# Initialize a requests session with retry strategy
session = requests.Session()
retry_strategy = Retry(
    total=5,
    read=5,      # Okuma zaman aşımı için yeniden deneme
    connect=5,   # Bağlantı hataları için yeniden deneme
    backoff_factor=0.3,
    status_forcelist=(500, 502, 504),
    allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]  # Yeniden denenecek HTTP yöntemleri
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Set custom headers
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.3'
})


def download_pdf(pdf_url, download_path):
    """
    Belirtilen URL'den PDF indirir ve belirtilen yola kaydeder.

    Args:
        pdf_url (str): İndirilecek PDF'nin URL'si.
        download_path (str): PDF'nin kaydedileceği yerel yol.

    Returns:
        bool: İndirme başarılıysa True, aksi halde False.
    """
    try:
        logger.info(f"URL'den PDF indiriliyor: {pdf_url}")
        response = session.get(pdf_url, stream=True, verify=False)
        response.raise_for_status()
        with open(download_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"PDF başarıyla indirildi: {download_path}")
        return True
    except requests.exceptions.RequestException as e:
        logger.error(f"PDF indirirken istek hatası: {e}")
    except Exception as e:
        logger.error(f"PDF indirirken beklenmeyen hata: {e}")
    return False


def extract_text_from_pdf(pdf_bytes):
    """
    PDF'den metin çıkarır.

    Args:
        pdf_bytes (BytesIO): PDF dosyasının byte hali.

    Returns:
        str: PDF'den çıkarılan metin.
    """
    try:
        logger.debug("PDF'den metin çıkarma başlatılıyor.")
        pdf_bytes.seek(0)
        with pdfplumber.open(pdf_bytes) as pdf:
            text = ""
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    logger.debug(f"Sayfa {page_num} metni çıkarıldı.")
        logger.info("PDF'den metin başarıyla çıkarıldı.")
        return text
    except Exception as e:
        logger.error(f"PDF'den metin çıkarılırken hata: {e}")
        return ""


def extract_images_from_pdf(pdf_path, images_dir):
    """
    PDF'den tüm resimleri çıkarır.

    Args:
        pdf_path (str): PDF dosyasının yolu.
        images_dir (str): Çıkarılan resimlerin kaydedileceği dizin.

    Returns:
        list: Çıkarılan resim dosyalarının yolları.
    """
    try:
        logger.info(f"PDF'den resimler çıkarılıyor: {pdf_path}")
        doc = fitz.open(pdf_path)
        os.makedirs(images_dir, exist_ok=True)
        image_paths = []
        for page_number in range(len(doc)):
            page = doc[page_number]
            images = page.get_images(full=True)
            for img_index, img in enumerate(images):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_name = f"page_{page_number+1}img{img_index+1}.{image_ext}"
                image_path = os.path.join(images_dir, image_name)
                with open(image_path, 'wb') as img_file:
                    img_file.write(image_bytes)
                image_paths.append(image_path)
                logger.debug(f"Resim çıkarıldı: {image_path}")
        logger.info(f"PDF'den {len(image_paths)} resim çıkarıldı.")
        return image_paths
    except Exception as e:
        logger.error(f"PDF'den resim çıkarılırken hata: {e}")
        return []


def create_pdf_with_images(images, output_pdf_path):
    """
    Resimleri içeren yeni bir PDF oluşturur.

    Args:
        images (list): Resim dosyalarının yolları.
        output_pdf_path (str): Yeni PDF'nin kaydedileceği yol.

    Returns:
        bool: PDF oluşturma başarılıysa True, aksi halde False.
    """
    try:
        logger.info(f"{len(images)} resimle yeni PDF oluşturuluyor: {output_pdf_path}")
        c = canvas.Canvas(output_pdf_path, pagesize=letter)
        for img_path in images:
            try:
                img = Image.open(img_path)
                img_width, img_height = img.size
                dpi = img.info.get('dpi', (72, 72))
                width_pt = img_width * 72.0 / dpi[0]
                height_pt = img_height * 72.0 / dpi[1]
                c.setPageSize((width_pt, height_pt))
                c.drawImage(img_path, 0, 0, width=width_pt, height=height_pt)
                c.showPage()
                logger.debug(f"Resim PDF'ye eklendi: {img_path}")
            except Exception as img_e:
                logger.error(f"Resim PDF'ye eklenirken hata: {img_path}, Hata: {img_e}")
        c.save()
        logger.info(f"Yeni PDF başarıyla oluşturuldu: {output_pdf_path}")
        return True
    except Exception as e:
        logger.error(f"Resimlerle PDF oluşturulurken hata: {e}")
        return False


def image_to_base64(image_path):
    """
    Resim dosyasını Base64 stringine dönüştürür.

    Args:
        image_path (str): Resim dosyasının yolu.

    Returns:
        str: Base64 kodlu resim.
    """
    try:
        with open(image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        logger.error(f"Resim Base64'e dönüştürülürken hata: {e}")
        return ""


def perform_ocr_on_pdf(pdf_path, images_dir, tc_image_save_dir):
    """
    PDF üzerindeki resimler üzerinde OCR gerçekleştirir ve geçerli TC Kimlik Numaralarını tespit eder.

    Args:
        pdf_path (str): PDF dosyasının yolu.
        images_dir (str): Çıkarılan resimlerin bulunduğu dizin.
        tc_image_save_dir (str): TC bulunan resimlerin kaydedileceği dizin.

    Returns:
        tuple: (TC bulundu mu, TC sayısı, Base64 resim)
    """
    try:
        logger.info(f"OCR işlemi başlatılıyor: {pdf_path}")
        reader = easyocr.Reader(['tr'], gpu=USE_GPU)
        doc = fitz.open(pdf_path)
        for page_number in range(len(doc)):
            page = doc.load_page(page_number)
            pix = page.get_pixmap()
            img_data = pix.tobytes("ppm")
            img = Image.open(BytesIO(img_data))
            ocr_result = reader.readtext(np.array(img), detail=0, paragraph=True)
            page_text = "\n".join(ocr_result)
            logger.debug(f"Sayfa {page_number+1} için OCR tamamlandı.")
            # Geçerli TC Kimlik Numaralarını bul
            numbers = re.findall(r'(?<!\d)\d{11}(?!\d)', page_text)
            valid_tc = [num for num in numbers if is_valid_tc_kimlik_no(num)]
            if valid_tc:
                tc_count = len(valid_tc)
                logger.info(f"Sayfa {page_number+1}'de {tc_count} geçerli TC Kimlik Numarası bulundu: {valid_tc}")
                # İlgili resmi kaydet
                image_filename = f"tc_found_page_{page_number+1}.png"
                image_save_path = os.path.join(tc_image_save_dir, image_filename)
                original_image_path = os.path.join(images_dir, f"page_{page_number+1}_img_1.png")  # Varsayılan olarak ilk resmi alıyoruz
                if os.path.exists(original_image_path):
                    shutil.copy(original_image_path, image_save_path)
                    logger.info(f"TC bulunan resim kaydedildi: {image_save_path}")
                else:
                    # Eğer spesifik bir resim yoksa, PDF sayfasını kaydedebiliriz
                    page_image_save_path = os.path.join(tc_image_save_dir, f"tc_page_{page_number+1}.png")
                    img.save(page_image_save_path)
                    logger.info(f"TC bulunan sayfa resmi kaydedildi: {page_image_save_path}")
                    image_save_path = page_image_save_path  # Base64 için doğru dosya yolunu belirle

                # Base64 encoding of the saved image
                base64_image = image_to_base64(image_save_path)
                return True, tc_count, base64_image
        logger.info("OCR işlemi tamamlandı ancak geçerli TC Kimlik Numarası bulunamadı.")
        return False, 0, ""
    except Exception as e:
        logger.error(f"OCR işlemi sırasında hata: {e}")
        return False, 0, ""


def extract_text_and_find_tc(pdf_path):
    """
    PDF'den metin çıkarır ve geçerli TC Kimlik Numaralarını bulur.

    Args:
        pdf_path (str): PDF dosyasının yolu.

    Returns:
        list: Geçerli TC Kimlik Numaralarının listesi.
    """
    try:
        logger.info(f"PDF'den metin çıkarılıyor: {pdf_path}")
        with open(pdf_path, 'rb') as f:
            pdf_bytes = BytesIO(f.read())
        text = extract_text_from_pdf(pdf_bytes)
        numbers = re.findall(r'(?<!\d)\d{11}(?!\d)', text)
        valid_tc = [num for num in numbers if is_valid_tc_kimlik_no(num)]
        logger.info(f"PDF'de {len(valid_tc)} geçerli TC Kimlik Numarası bulundu.")
        return valid_tc
    except Exception as e:
        logger.error(f"PDF'den metin çıkarılırken hata: {e}")
        return []


def perform_ocr_if_needed(pdf_path, images_dir, tc_image_save_dir):
    """
    OCR yapılarak geçerli TC Kimlik Numaralarını bulur.

    Args:
        pdf_path (str): PDF dosyasının yolu.
        images_dir (str): Resimlerin çıkarılacağı dizin.
        tc_image_save_dir (str): TC bulunan resimlerin kaydedileceği dizin.

    Returns:
        tuple: (TC bulundu mu, TC sayısı, Base64 resim)
    """
    images = extract_images_from_pdf(pdf_path, images_dir)
    if images:
        logger.info(f"{len(images)} resim PDF'den çıkarıldı. OCR işlemi başlatılıyor.")
        images_pdf_path = os.path.join(images_dir, "images_document.pdf")
        if create_pdf_with_images(images, images_pdf_path):
            return perform_ocr_on_pdf(images_pdf_path, images_dir, tc_image_save_dir)
        else:
            logger.error("Resimlerle PDF oluşturulamadı. OCR işlemi yapılamadı.")
    else:
        logger.info("PDF'de çıkarılacak resim bulunamadı. OCR işlemi yapılamadı.")
    return False, 0, ""


def validate_and_process(url, ocr_enabled=True, verify_ssl=True):
    """
    Tek bir PDF URL'sini doğrular ve işler, geçerli TC Kimlik Numaralarını tespit eder.

    Args:
        url (str): İşlenecek PDF URL'si.
        ocr_enabled (bool): OCR yapılıp yapılmayacağı.
        verify_ssl (bool): SSL sertifika doğrulaması.

    Returns:
        dict: URL, durum, alan adı, TC sayısı, OCR durumu ve resim bilgisi.
    """
    try:
        # URL'nin şemasını kontrol et ve düzelt
        url = ensure_url_scheme(url)
        if not url:
            logger.error(f"URL geçersiz: {url}. Atlanıyor.")
            return {"domain": "Unknown", "url": url, "status": "Invalid URL", "tc_count": 0}

        logger.info(f"İşleniyor: {url} | SSL Doğrulama: {'Etkin' if verify_ssl else 'Devre Dışı'}")
        
        # Adım 1: PDF'yi indir
        temp_dir = os.path.join(PROJECT_ROOT, "temp_processing")
        os.makedirs(temp_dir, exist_ok=True)
        pdf_filename = os.path.join(temp_dir, "downloaded.pdf")
        if not download_pdf(url, pdf_filename):
            logger.warning(f"URL: {url} - İndirme başarısız.")
            return {"domain": "Unknown", "url": url, "status": "Download Error", "tc_count": 0}

        parsed_url = urlparse(url)
        domain = get_main_domain(url)

        pdf_basename = os.path.basename(parsed_url.path)
        pdf_name, _ = os.path.splitext(pdf_basename)
        if not pdf_name:
            pdf_name = "downloaded"

        # Adım 2: PDF'den metin çıkar ve TC bul
        valid_tc = extract_text_and_find_tc(pdf_filename)

        if valid_tc:
            tc_count = len(valid_tc)
            logger.info(f"URL: {url} - Metin içerisinden {tc_count} geçerli TC Kimlik Numarası bulundu.")
            return {
                "domain": domain,
                "url": url,
                "status": "POSITIVE",
                "tc_count": tc_count,
                "ocr_enabled": False,
                "image": ""
            }

        if ocr_enabled:
            # Adım 3: OCR ile TC bul
            images_dir = os.path.join(temp_dir, "extracted_images")
            tc_image_save_dir = os.path.join(temp_dir, "tc_images")
            os.makedirs(tc_image_save_dir, exist_ok=True)
            tc_found, tc_count, base64_image = perform_ocr_if_needed(pdf_filename, images_dir, tc_image_save_dir)
            if tc_found:
                return {
                    "domain": domain,
                    "url": url,
                    "status": "POSITIVE",
                    "tc_count": tc_count,
                    "ocr_enabled": True,
                    "image": base64_image
                }

        logger.info(f"URL: {url} - Hem metin hem de OCR ile geçerli TC Kimlik Numarası bulunamadı.")
        return {"domain": domain, "url": url, "status": "Negative", "tc_count": 0, "ocr_enabled": False, "image": ""}

    except requests.exceptions.RequestException as e:
        logger.error(f"URL: {url} - Ağ hatası: {e}")
        return {"domain": "Unknown", "url": url, "status": "Network Error", "tc_count": 0, "ocr_enabled": False, "image": ""}
    except Exception as e:
        logger.error(f"URL: {url} - Beklenmeyen hata: {e}")
        return {"domain": "Unknown", "url": url, "status": "Processing Error", "tc_count": 0, "ocr_enabled": False, "image": ""}


def process_pdfs(urls, ocr_enabled=True):
    """
    PDF URL'lerini işler ve geçerli TC Kimlik Numaralarını tespit eder.

    Args:
        urls (list): İşlenecek PDF URL'lerinin listesi.
        ocr_enabled (bool): OCR yapılıp yapılmayacağı.

    Returns:
        dict: İşlem sonuçlarını içeren sözlük.
    """
    results = {}

    for url in urls:
        url = url.strip()
        if not url:
            continue  # Boş satırları atla

        # SSL doğrulamasını belirle
        if "msb.gov.tr" in url or "tarimorman.gov.tr" in url:
            verify_ssl = False
            logger.debug(f"URL: {url} için SSL doğrulaması devre dışı bırakıldı.")
        else:
            verify_ssl = True
            logger.debug(f"URL: {url} için SSL doğrulaması etkin.")

        # URL'yi işle
        result = validate_and_process(url, ocr_enabled, verify_ssl)

        domain = result.get("domain", "Unknown")
        status = result.get("status", "Unknown")

        if domain not in results:
            results[domain] = {
                "Domain": domain,
                "PDF": {
                    "POSITIVE": {
                        "Urls": [],
                        "TC_Count": 0
                    },
                    "Negative": {
                        "Urls": []
                    },
                    "Error": {
                        "Urls": [],
                        "Statuses": []
                    }
                }
            }

        if status == "POSITIVE":
            results[domain]["PDF"]["POSITIVE"]["Urls"].append({
                "url": result["url"],
                "tc_count": result["tc_count"],
                "ocr_enabled": result.get("ocr_enabled", False),
                "image": result.get("image", "")
            })
            results[domain]["PDF"]["POSITIVE"]["TC_Count"] += result.get("tc_count", 0)
        elif status == "Negative":
            results[domain]["PDF"]["Negative"]["Urls"].append({
                "url": result["url"]
            })
        else:
            # "Download Error" veya "Processing Error" gibi durumlar için
            results[domain]["PDF"]["Error"]["Urls"].append({
                "url": result["url"]
            })
            results[domain]["PDF"]["Error"]["Statuses"].append({
                "status": status
            })

    return results