# detectors/XLS_XLSX_Detector.py

import os
import re
import sys
import ssl
import json
import logging
import argparse
from io import BytesIO
from urllib.parse import urlparse
import requests
import pandas as pd
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

from detectors.util import (
    download_file,
    is_valid_tc_kimlik_no,
    get_main_domain,
    ensure_url_scheme,
    is_valid_url
)
from dorking_tool.logger_setup import setup_logger

# SSL sertifikalarını doğrulamamak için ayarlar
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Yüklemeleri ve yardımcı modülleri içe aktar
load_dotenv()

# Merkezi logger'ı yapılandır
logger = setup_logger()

# Constants
THREAD_COUNT = int(os.getenv('xls_detector_THREAD_COUNT', 4))  # Örnek: 4 çalışan
USE_GPU = os.getenv('xls_detector_USE_GPU', 'False').lower() in ['true', '1', 't']

# Proje kök dizinini belirle
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

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


def extract_numbers_from_df(df):
    """
    DataFrame'deki tüm hücrelerden 11 haneli sayıları (TC Kimlik No) çıkarır.

    Args:
        df (pd.DataFrame): Excel dosyasından okunan DataFrame.

    Returns:
        list: Çıkarılan benzersiz 11 haneli sayıların listesi.
    """
    pattern = re.compile(r'(?<!\d)\d{11}(?!\d)')
    numbers = set()
    for column in df.columns:
        for item in df[column].astype(str):
            found = pattern.findall(item)
            for num in found:
                numbers.add(num)
    return list(numbers)


def analyze_file(file_path):
    """
    Excel dosyasını analiz eder ve geçerli TC Kimlik No sayılarını sayar.

    Args:
        file_path (str): Excel dosyasının yolu.

    Returns:
        int: Geçerli TC Kimlik No sayı adedi.
    """
    try:
        # Dosya uzantısını kontrol et
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == '.xlsx':
            engine = 'openpyxl'
        elif file_extension == '.xls':
            engine = 'xlrd'
        else:
            raise ValueError("Desteklenmeyen dosya formatı. Sadece .xls ve .xlsx desteklenmektedir.")

        # Tüm sayfaları oku
        xls = pd.ExcelFile(file_path, engine=engine)
        all_numbers = set()
        for sheet in xls.sheet_names:
            df = pd.read_excel(xls, sheet_name=sheet, dtype=str, engine=engine)
            numbers = extract_numbers_from_df(df)
            all_numbers.update(numbers)

        valid_numbers = [num for num in all_numbers if is_valid_tc_kimlik_no(num)]

        valid_tc_count = len(valid_numbers)
        logger.info(f"Dosya analiz edildi: {file_path}, Geçerli TC Kimlik No sayısı: {valid_tc_count}")
        return valid_tc_count  # Geçerli sayı adedini döner

    except Exception as e:
        logger.error(f"Dosya analizi sırasında hata oluştu: {file_path}, Hata: {e}")
        return 0


def handle_file_extension(url):
    """
    URL'nin dosya uzantısını kontrol eder ve döner.
    Eğer dosya uzantısı yoksa, XLSX olarak varsayar.

    Args:
        url (str): Dosyanın URL'si.

    Returns:
        str: Dosya uzantısı (.xls veya .xlsx) veya '.xlsx' varsayılanı.
    """
    parsed_url = urlparse(url)
    file_extension = os.path.splitext(parsed_url.path)[1].lower()
    if file_extension in ['.xls', '.xlsx']:
        return file_extension
    else:
        logger.warning(f"URL: {url} dosya uzantısı içermiyor. Varsayılan olarak '.xlsx' uzantısı atanıyor.")
        return '.xlsx'


def download_and_process_file(url):
    """
    URL'den dosyayı indirir ve doğru uzantıyı ekler.

    Args:
        url (str): Dosyanın URL'si.

    Returns:
        str: İndirilen dosyanın yolu veya None.
    """
    try:
        file_extension = handle_file_extension(url)
        downloaded_file = download_file(url, DOWNLOAD_DIR)
        if not downloaded_file:
            logger.warning(f"Dosya indirilemedi: {url}")
            return None

        # İndirilen dosyanın uzantısını kontrol et ve gerekirse ekle
        if not downloaded_file.endswith(file_extension):
            new_file_path = downloaded_file + file_extension
            os.rename(downloaded_file, new_file_path)
            logger.debug(f"Dosya uzantısı eklendi: {new_file_path}")
            downloaded_file = new_file_path

        logger.debug(f"Dosya indirildi: {downloaded_file}")
        return downloaded_file

    except Exception as e:
        logger.error(f"Dosya indirme veya işleme sırasında hata: {url}, Hata: {e}")
        return None


def validate_and_process(url, verify_ssl=True):
    """
    Tek bir XLS/XLSX URL'sini indirir ve analiz eder.

    Args:
        url (str): İşlenecek XLS/XLSX URL'si.
        verify_ssl (bool): SSL sertifika doğrulaması.

    Returns:
        dict: URL, durum, alan adı ve TC sayısı bilgilerini içeren sözlük.
    """
    try:
        # URL'nin şemasını kontrol et ve düzelt
        url = ensure_url_scheme(url)
        if not url:
            logger.error(f"URL geçersiz: {url}. Atlanıyor.")
            return {"domain": "Unknown", "url": url, "status": "Invalid URL", "tc_count": 0}

        logger.info(f"İşleniyor: {url} | SSL Doğrulama: {'Etkin' if verify_ssl else 'Devre Dışı'}")
        # Dosyayı indir ve işleme
        downloaded_file = download_and_process_file(url)
        if not downloaded_file:
            logger.warning(f"İndirilen dosya yok: {url}")
            return {"domain": "Unknown", "url": url, "status": "Download Failed", "tc_count": 0}

        # Alan adını çıkar
        domain = get_main_domain(url)

        # Dosyayı analiz et
        valid_tc_count = analyze_file(downloaded_file)

        # Durumu belirle
        status = "POSITIVE" if valid_tc_count > 0 else "Negative"

        result = {
            "domain": domain,
            "url": url,
            "status": status,
            "tc_count": valid_tc_count
        }

        logger.info(f"URL: {url} - Durum: {status} - Geçerli TC Kimlik No sayısı: {valid_tc_count}")

        # İndirilen dosyayı sil
        try:
            os.remove(downloaded_file)
            logger.info(f"Geçici dosya silindi: {downloaded_file}")
        except OSError as e:
            logger.error(f"Dosya silinirken hata oluştu: {downloaded_file}, Hata: {e}")

        return result

    except Exception as e:
        logger.error(f"URL işlenirken beklenmeyen hata: {url}, Hata: {e}")
        return {"domain": "Unknown", "url": url, "status": "Processing Error", "tc_count": 0}


def process_xls_xlsx(urls, verify_ssl_list=None):
    """
    XLS/XLSX URL'lerini işler ve sonuçları döner.

    Args:
        urls (list): İşlenecek XLS/XLSX URL'lerinin listesi.
        verify_ssl_list (list, optional): Her URL için SSL doğrulamasının etkin olup olmadığını belirten liste.

    Returns:
        dict: İşlem sonuçlarını içeren sözlük.
    """
    results = {}

    for idx, url in enumerate(urls):
        url = url.strip()
        if not url:
            continue  # Boş satırları atla

        # SSL doğrulamasını belirle
        if verify_ssl_list and idx < len(verify_ssl_list):
            verify_ssl = verify_ssl_list[idx]
        else:
            if "msb.gov.tr" in url or "tarimorman.gov.tr" in url:
                verify_ssl = False
            else:
                verify_ssl = True

        logger.debug(f"URL: {url} için SSL doğrulaması: {'Etkin' if verify_ssl else 'Devre Dışı'}")

        # URL'yi işle
        result = validate_and_process(url, verify_ssl)

        domain = result.get("domain", "Unknown")
        status = result.get("status", "Unknown")

        if domain not in results:
            results[domain] = {
                "XLS_XLSX": {
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
            results[domain]["XLS_XLSX"]["POSITIVE"]["Urls"].append({
                "url": result["url"],
                "tc_count": result["tc_count"]
            })
            results[domain]["XLS_XLSX"]["POSITIVE"]["TC_Count"] += result.get("tc_count", 0)
        elif status == "Negative":
            results[domain]["XLS_XLSX"]["Negative"]["Urls"].append({
                "url": result["url"]
            })
        else:
            # "Download Failed" veya "Processing Error" gibi durumlar için
            results[domain]["XLS_XLSX"]["Error"]["Urls"].append({
                "url": result["url"]
            })
            results[domain]["XLS_XLSX"]["Error"]["Statuses"].append({
                "status": status
            })

    return results


def main():
    """
    Ana fonksiyon. Komut satırı argümanlarını parse eder, URL'leri işler ve sonuçları çıktılar.
    """
    parser = argparse.ArgumentParser(description='XLS/XLSX URL analiz aracı.')
    parser.add_argument('input_json', type=str, help='URL\'leri içeren JSON dosyasının yolu.')
    args = parser.parse_args()

    input_json = args.input_json

    # JSON dosyasından URL'leri oku
    try:
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, dict):
                urls = data.get('urls', [])
            elif isinstance(data, list):
                urls = data
            else:
                urls = []
                logger.error(f"Beklenmeyen JSON formatı: {input_json}")
        logger.info(f"{input_json} dosyasından {len(urls)} URL yüklendi.")
    except Exception as e:
        logger.error(f"JSON dosyası okunamadı: {input_json}, Hata: {e}")
        sys.exit(1)

    if not urls:
        logger.error("Giriş JSON dosyasında URL bulunamadı.")
        sys.exit(1)

    # URL'leri işle
    results = process_xls_xlsx(urls)

    # Sonuçları JSON formatında yazdır
    print(json.dumps(results, ensure_ascii=False, indent=4))

    logger.info("XLS/XLSX URL analizi tamamlandı.")


if __name__ == "__main__":
    main()