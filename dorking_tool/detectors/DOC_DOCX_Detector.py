# detectors/DOC_DOCX_Detector.py

import os
import sys
import ssl
import json
import logging
import argparse
from urllib.parse import urlparse
from detectors.util import (
    download_file,
    is_valid_tc_kimlik_no,
    get_main_domain,
    ensure_url_scheme,
    is_valid_url
)
import subprocess
import docx2txt
import re

# SSL sertifika doğrulamasını devre dışı bırak
ssl._create_default_https_context = ssl._create_unverified_context

# Proje kök dizinini ayarla
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOWNLOAD_DIR = os.path.join(PROJECT_ROOT, "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# Logger yapılandırması (Sadece konsol çıktısı)
logger = logging.getLogger('docx_doc_detector')
logger.setLevel(logging.DEBUG)
logger.propagate = False

formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s')

# StreamHandler'ı kaldırdık ve sadece merkezi logger'ı kullanıyoruz
# Eğer konsol çıktısı istiyorsanız, aşağıdaki satırları ekleyebilirsiniz:
# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.DEBUG)
# console_handler.setFormatter(formatter)
# logger.addHandler(console_handler)


def extract_text_from_docx(file_path):
    """
    DOCX dosyasından metin çıkarır.
    
    Args:
        file_path (str): DOCX dosyasının yolu.
    
    Returns:
        str: Çıkarılan metin.
    """
    try:
        text = docx2txt.process(file_path)
        if not text.strip():
            logger.warning(f"Metin çıkarılamadı veya dosya boş: {file_path}")
        else:
            logger.debug(f"DOCX dosyasından metin çıkarıldı: {file_path}")
        return text
    except Exception as e:
        logger.error(f"DOCX dosyası okunamadı: {file_path}, Hata: {e}")
        return ""


def extract_text_from_doc(file_path):
    """
    DOC dosyasından metin çıkarır 'antiword' aracını kullanarak.
    
    Args:
        file_path (str): DOC dosyasının yolu.
    
    Returns:
        str: Çıkarılan metin.
    """
    try:
        result = subprocess.run(['antiword', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            logger.error(f"Antiword hatası: {result.stderr}")
            return ""
        text = result.stdout
        logger.debug(f"DOC dosyasından metin çıkarıldı: {file_path}")
        return text
    except FileNotFoundError:
        logger.error("Antiword aracı bulunamadı. Lütfen 'antiword' yükleyin.")
        return ""
    except Exception as e:
        logger.error(f"DOC dosyası okunamadı: {file_path}, Hata: {e}")
        return ""


def extract_numbers_from_text(text):
    """
    Metinden 11 haneli sayılar (TC Kimlik No) çıkarır.
    
    Args:
        text (str): Metin.
    
    Returns:
        list: Çıkarılan 11 haneli sayılar.
    """
    pattern = re.compile(r'(?<!\d)\d{11}(?!\d)')
    numbers = set(pattern.findall(text))
    return list(numbers)


def analyze_file(file_path):
    """
    DOC/DOCX dosyasını analiz eder ve geçerli TC Kimlik No sayısını döner.
    
    Args:
        file_path (str): DOC/DOCX dosyasının yolu.
    
    Returns:
        int: Geçerli TC Kimlik No sayısı.
    """
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        # Dosya uzantısına göre metin çıkarma
        if file_extension == '.docx':
            text = extract_text_from_docx(file_path)
        elif file_extension == '.doc':
            text = extract_text_from_doc(file_path)
        else:
            raise ValueError("Desteklenmeyen dosya formatı. Sadece .doc ve .docx desteklenmektedir.")
        
        if not text.strip():
            logger.warning(f"Dosya boş veya metin çıkarılamadı: {file_path}")
            return 0

        # 11 haneli sayıları çıkar ve geçerliliğini kontrol et
        numbers = extract_numbers_from_text(text)
        valid_numbers = [num for num in numbers if is_valid_tc_kimlik_no(num)]
        valid_tc_count = len(valid_numbers)

        logger.info(f"Dosya analiz edildi: {file_path}, Geçerli TC Kimlik No sayısı: {valid_tc_count}")
        return valid_tc_count

    except Exception as e:
        logger.error(f"Dosya analizi sırasında hata oluştu: {file_path}, Hata: {e}")
        return 0


def handle_file_extension(url):
    """
    URL'nin dosya uzantısını kontrol eder ve döner.
    Eğer dosya uzantısı yoksa, DOCX olarak varsayar.
    
    Args:
        url (str): Dosyanın URL'si.
    
    Returns:
        str: Dosya uzantısı (.doc veya .docx) veya '.docx' varsayılanı.
    """
    parsed_url = urlparse(url)
    file_extension = os.path.splitext(parsed_url.path)[1].lower()
    if file_extension in ['.doc', '.docx']:
        return file_extension
    else:
        logger.warning(f"URL: {url} dosya uzantısı içermiyor. Varsayılan olarak '.docx' uzantısı atanıyor.")
        return '.docx'


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
    Tek bir DOC/DOCX URL'sini indirir ve analiz eder.
    
    Args:
        url (str): İşlenecek URL.
        verify_ssl (bool): SSL sertifika doğrulaması.
    
    Returns:
        dict: İşlem sonucu bilgileri.
    """
    try:
        # URL şemasını kontrol et ve düzelt
        url = ensure_url_scheme(url)
        if not url:
            logger.error(f"URL geçersiz: {url}")
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


def process_doc_docx(urls):
    """
    DOC/DOCX URL'lerini işler ve sonuçları döner.
    
    Args:
        urls (list): İşlenecek URL listesi.
    
    Returns:
        dict: İşlem sonuçları.
    """
    results = {}

    for url in urls:
        url = url.strip()
        if not url:
            continue  # Boş satırları atla

        # SSL doğrulamasını belirle
        if "msb.gov.tr" in url or "tarimorman.gov.tr" in url:
            verify_ssl = False
            logger.debug(f"SSL doğrulaması devre dışı: {url}")
        else:
            verify_ssl = True
            logger.debug(f"SSL doğrulaması etkin: {url}")

        # URL'yi işle
        result = validate_and_process(url, verify_ssl)

        domain = result.get("domain", "Unknown")
        status = result.get("status", "Unknown")

        if domain not in results:
            results[domain] = {
                "DOC_DOCX": {
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
            results[domain]["DOC_DOCX"]["POSITIVE"]["Urls"].append({
                "url": result["url"],
                "tc_count": result["tc_count"]
            })
            results[domain]["DOC_DOCX"]["POSITIVE"]["TC_Count"] += result.get("tc_count", 0)
        elif status == "Negative":
            results[domain]["DOC_DOCX"]["Negative"]["Urls"].append({
                "url": result["url"]
            })
        else:
            results[domain]["DOC_DOCX"]["Error"]["Urls"].append({
                "url": result["url"]
            })
            results[domain]["DOC_DOCX"]["Error"]["Statuses"].append({
                "status": status
            })

    return results


def main():
    """
    Ana fonksiyon. Komut satırı argümanlarını parse eder, URL'leri işler ve sonuçları çıktılar.
    """
    parser = argparse.ArgumentParser(description='DOC/DOCX URL analiz aracı.')
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
    results = process_doc_docx(urls)

    # Sonuçları JSON formatında yazdır
    print(json.dumps(results, ensure_ascii=False, indent=4))

    logger.info("DOC/DOCX URL analizi tamamlandı.")


if __name__ == "__main__":
    main()